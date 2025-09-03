import jax
import flax
import optax
import functools
import numpy as np
import ml_collections
import jax.numpy as jnp

from typing import Any
from copy import deepcopy

from M_utils.flax_utils import ModuleDict, TrainState
from M_utils.encoders import encoder_modules, GCEncoder
from M_utils.networks import GCPhiValue, GCValue, GCActor

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class GASAgent(flax.struct.PyTreeNode):
    """Graph-Assisted Stitching (GAS) agent"""
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config,):
        """
        Create a new agent.
        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        ex_tdr_dim = np.zeros((1, config['tdr_dim']))
        action_dim = ex_actions.shape[-1]

        # Define encoders
        encoders = dict()
        if config['encoder'] == 'not_used':
            print("Using state-based observations (no encoder).")
        elif config['encoder'] in encoder_modules:
            print(f"Using pixel-based observations with encoder: {config['encoder']}")
            encoder_module = encoder_modules[config['encoder']]
            encoders['tdr_value'] = GCEncoder(state_encoder=encoder_module())
            encoders['value'] = GCEncoder(state_encoder=encoder_module())
            encoders['critic'] = GCEncoder(state_encoder=encoder_module())
            encoders['actor'] = GCEncoder(state_encoder=encoder_module())
        else:
            raise ValueError(f"Unknown encoder: {config['encoder']}")
            
        # Define TDR value, value, critic and actor networks.
        tdr_value_def = GCPhiValue(ensemble=True, hidden_dims=config['tdr_value_hidden_dims'], tdr_dim=config['tdr_dim'], layer_norm=config['layer_norm'], gc_encoder=encoders.get('tdr_value'),)
        value_def = GCValue(ensemble=False, hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'], gc_encoder=encoders.get('value'),)
        critic_def = GCValue(ensemble=True, hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'], gc_encoder=encoders.get('critic'),)
        actor_def = GCActor(hidden_dims=config['actor_hidden_dims'], action_dim=action_dim, final_fc_init_scale=config['final_fc_init_scale'], 
                            state_dependent_std=config['state_dependent_std'], const_std=config['const_std'], gc_encoder=encoders.get('actor'), 
                            log_std_min=config['log_std_min'], log_std_max=config['log_std_max'], tanh_squash=config['tanh_squash'])

        network_info = dict(
            tdr_value=(tdr_value_def, (ex_observations, ex_goals, False, False)),
            target_tdr_value=(deepcopy(tdr_value_def), (ex_observations, ex_goals, False, False)),
            value=(value_def, (ex_observations, ex_tdr_dim , None, True)),
            critic=(critic_def, (ex_observations, ex_tdr_dim , ex_actions, True)),
            target_critic=(deepcopy(critic_def), (ex_observations, ex_tdr_dim , ex_actions, True)),
            actor=(actor_def, (ex_observations, ex_tdr_dim, 1.0, True)),
        )
        
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']
        network_tx = optax.adam(learning_rate=config['lr'])
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_tdr_value'] = params['modules_tdr_value']
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))
    
    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def tdr_value_loss(self, batch, grad_params):
        """Compute the TDR value loss."""
        next_v1, next_v2 = self.network.select('target_tdr_value')(batch['next_observations'], batch['tdr_value_goals'], goal_encoded=False, get_phi=False)
        
        next_v = jnp.minimum(next_v1, next_v2)
        q = batch['tdr_rewards'] + self.config['discount'] * batch['tdr_masks'] * next_v
        v1_t, v2_t = self.network.select('target_tdr_value')(batch['observations'], batch['tdr_value_goals'], goal_encoded=False, get_phi=False)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['tdr_rewards'] + self.config['discount'] * batch['tdr_masks'] * next_v1
        q2 = batch['tdr_rewards'] + self.config['discount'] * batch['tdr_masks'] * next_v2
        v1, v2 = self.network.select('tdr_value')(batch['observations'], batch['tdr_value_goals'], goal_encoded=False, get_phi=False, params=grad_params)
        v = (v1 + v2) / 2
        
        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['tdr_expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['tdr_expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {'value_loss': value_loss, 'v max': v.max(), 'v min': v.min(), 'v mean': v.mean(), 
                            'abs adv mean': jnp.abs(adv).mean(), 'adv mean': adv.mean(), 'adv max': adv.max(), 'adv min': adv.min(), 'accept prob': (adv >= 0).mean(),}

    def value_loss(self, batch, grad_params):
        """Compute the value loss."""
        q1, q2 = self.network.select('target_critic')(batch['observations'], batch['value_skills'], batch['actions'], goal_encoded=True)
        q = jnp.minimum(q1, q2)
        
        v = self.network.select('value')(batch['observations'], batch['value_skills'], None, goal_encoded=True, params=grad_params)
        
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()
        
        return value_loss, {'value_loss': value_loss, 'v_mean': v.mean(), 'v_max': v.max(), 'v_min': v.min(),}

    def critic_loss(self, batch, grad_params):
        """Compute the critic loss."""
        next_v = self.network.select('value')(batch['next_observations'], batch['value_skills'], None, goal_encoded=True)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        
        q1, q2 = self.network.select('critic')(batch['observations'], batch['value_skills'], batch['actions'], goal_encoded=True, params=grad_params)
        
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()
        
        return critic_loss, {'critic_loss': critic_loss, 'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min(),}

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss."""
        dist = self.network.select('actor')(batch['observations'], batch['actor_skills'], temperature=1.0, goal_encoded=True, params=grad_params)
        
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
        q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_skills'], q_actions, goal_encoded=True)
        q = jnp.minimum(q1, q2)
        # Normalize Q values by the absolute mean to make the loss scale invariant.
        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        
        log_prob = dist.log_prob(batch['actions'])
        bc_loss = -(self.config['alpha'] * log_prob).mean()

        actor_loss = q_loss + bc_loss

        return actor_loss, {'actor_loss': actor_loss, 'q_loss': q_loss, 'bc_loss': bc_loss, 'q_mean': q.mean(), 'q_abs_mean': jnp.abs(q).mean(), 'bc_log_prob': log_prob.mean(), 
                            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2), 'std': jnp.mean(dist.scale_diag),}

    @jax.jit
    def total_tdr_value_loss(self, batch, grad_params):
        """Compute the total TDR loss."""
        info = {}

        # TDR value
        tdr_value_loss, tdr_value_info = self.tdr_value_loss(batch, grad_params)
        
        # info
        for k, v in tdr_value_info.items():
            info[f'tdr value/{k}'] = v
                
        loss = tdr_value_loss 
        
        return loss, info
    
    @jax.jit
    def total_critic_actor_loss(self, batch, grad_params, rng=None):
        """Compute the total low-level policy loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng = jax.random.split(rng)
        epsilon=1e-10
        
        # value, critic
        batch_size, rep_dim = batch['phi_obs'].shape
        random_skills = jax.random.normal(rng, (batch_size, rep_dim))
        unnormalized_vg_norm = jnp.linalg.norm(random_skills, axis=1, keepdims=True) + epsilon
        batch['value_skills'] = random_skills / unnormalized_vg_norm 
        batch['rewards'] = ((batch['phi_next_obs'] - batch['phi_obs']) * batch['value_skills']).sum(axis=1)
        batch['masks'] = jnp.ones(batch_size)
        value_loss, value_info = self.value_loss(batch, grad_params)
        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        
        # actor
        unnormalized_ag_skills = batch['phi_actor_goals'] - batch['phi_obs']
        unnormalized_ag_norm = jnp.linalg.norm(unnormalized_ag_skills , axis=1, keepdims=True) + epsilon
        batch['actor_skills'] = unnormalized_ag_skills / unnormalized_ag_norm
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        
        # info
        for k, v in value_info.items():
            info[f'value/{k}'] = v
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
        
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']), 
                                                   self.network.params[f'modules_{module_name}'], self.network.params[f'modules_target_{module_name}'],)
        network.params[f'modules_target_{module_name}'] = new_target_params
        
    @jax.jit
    def tdr_update(self, batch):
        """Update the TDR and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)
        def tdr_loss_fn(grad_params):
            return self.total_tdr_value_loss(batch, grad_params)
        new_network, info = self.network.apply_loss_fn(loss_fn=tdr_loss_fn)
        
        self.target_update(new_network, 'tdr_value')
        
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def critic_actor_update(self, batch):
        """Update the low-level policy and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def critic_actor_loss_fn(grad_params):
            return self.total_critic_actor_loss(batch, grad_params, rng=rng)
        new_network, info = self.network.apply_loss_fn(loss_fn=critic_actor_loss_fn)
        
        self.target_update(new_network, 'critic')
        
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def get_phi(self, observations):
        """Get phi from the TDR."""
        phi = self.network.select('tdr_value')(observations, None, goal_encoded=False, get_phi=True)
        return phi
        
    @jax.jit
    def sample_actions(self, observations, goals=None, temperature=1.0, seed=None):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, goals, temperature=temperature, goal_encoded=True)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions


def get_config():
    config = ml_collections.ConfigDict(
        dict(            
            # Agent hyperparameters.
            agent_name='gas',                       # Agent name.
            encoder='not_used',                     # Visual encoder name ('not_used', 'impala_small', etc.).
            tdr_value_hidden_dims=(512, 512, 512),  # TDR network hidden dimensions.
            value_hidden_dims=(512, 512, 512),      # Value network hidden dimensions.
            actor_hidden_dims=(512, 512, 512),      # Actor network hidden dimensions.
            tdr_dim=32,                             # Latent dimension for phi.
            layer_norm=True,                        # Whether to use layer normalization.
            state_dependent_std=False,              # Whether to use state-dependent standard deviation for the actor.
            const_std=True,                         # Whether to use constant standard deviation for the actor.
            tanh_squash=False,                      # Whether to squash the action with tanh for the actor.
            log_std_min=-5,                         # Minimum value of log standard deviation for the actor.
            log_std_max=2,                          # Maximum value of log standard deviation for the actor.
            final_fc_init_scale=1e-2,               # Initial scale of the final fully-connected layer for the actor. 
            discount=0.995,                         # Discount factor.
            tdr_expectile=0.999,                    # TDR expectile.
            expectile=0.7,                          # Value expectile.
            alpha=1.0,                              # Temperature in BC coefficient.
            lr=3e-4,                                # Learning rate.
            tau=0.005,                              # Target network update rate.
        
            # Dataset hyperparameters.
            batch_size=1024,                        # Batch size.
            p_aug=0.0,                              # Probability of applying image augmentation.
            tdr_value_p_curgoal=0.0,                # Probability of using the current state as the TDR value goal.
            tdr_value_p_trajgoal=0.625,             # Probability of using a future state in the same trajectory as the TDR value goal.
            tdr_value_p_randomgoal=0.375,           # Probability of using a random state as the TDR value goal.
            tdr_value_geom_sample=True,             # Whether to use geometric sampling for future TDR value goals.
            way_steps=8,                            # Temporal Distance Threshold
        )
    )
    return config
