import distrax
import jax.numpy as jnp
import flax.linen as nn

from typing import Any, Optional, Sequence


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(cls, variable_axes={'params': 0}, split_rngs={'params': True}, in_axes=None, out_axes=out_axes, axis_size=num_qs, **kwargs,)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""
    def mode(self):
        return self.bijector.forward(self.distribution.mode())
    
    
class MLP(nn.Module):
    """
    Multi-layer perceptron.
    Attributes:
        hidden_dims: Hidden layer dimensions.
        kernel_init: Kernel initializer.
        activate_final: Whether to apply activation to the final layer.
        activations: Activation function.
        layer_norm: Whether to apply layer normalization.
    """
    hidden_dims: Sequence[int] = (512, 512, 512)
    kernel_init: Any = default_init()
    activate_final: bool = False
    activations: Any = nn.gelu
    layer_norm: bool = False
    
    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


class GCPhiValue(nn.Module):
    """
    TDR value function.
    This module can be used for TDR value V(s, g) functions.
    Attributes:
        ensemble: Whether to ensemble the TDR value function.
        hidden_dims: Hidden layer dimensions.
        tdr_dim: latent dimension.
        layer_norm: Whether to apply layer normalization.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """
    ensemble: bool = True
    hidden_dims: Sequence[int] = (512, 512, 512)
    tdr_dim: int = 32
    layer_norm: bool = True
    gc_encoder: nn.Module = None
    
    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        self.phi_net = mlp_module((*self.hidden_dims, self.tdr_dim), activate_final=False, layer_norm=self.layer_norm)
        
    def __call__(self, observations, goals=None, goal_encoded=False, get_phi=False):
        """
        Return the TDR value function.
        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            get_phi: Return the phi.
        """
        if self.gc_encoder is not None:
            observations = self.gc_encoder(observations, None, goal_encoded=goal_encoded)            
        phi_s = self.phi_net(observations)
        if get_phi:
            return phi_s[0]  
        else:
            if self.gc_encoder is not None:
                goals = self.gc_encoder(goals, None, goal_encoded=goal_encoded)
            phi_g = self.phi_net(goals)
            squared_dist = ((phi_s - phi_g) ** 2).sum(axis=-1)
            v = -jnp.sqrt(jnp.maximum(squared_dist, 1e-6))
            return v


class GCValue(nn.Module):
    """
    Goal-conditioned value/critic function.
    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.
    Attributes:
        ensemble: Whether to ensemble the value function.
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """
    ensemble: bool = True
    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True
    gc_encoder: nn.Module = None
    
    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        self.value_net = mlp_module((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals=None, actions=None, goal_encoded=True):
        """
        Return the value/critic function.
        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
            goal_encoded: Whether the goals are already encoded.
        """
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals, goal_encoded=goal_encoded)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)
        v = self.value_net(inputs).squeeze(-1)
        return v
    
    
class GCActor(nn.Module):
    """
    Goal-conditioned actor.
    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        gc_encoder: Optional GCEncoder module to encode the inputs.   
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
    """
    hidden_dims: Sequence[int] = (512, 512, 512)
    action_dim: int = 8
    final_fc_init_scale: float = 1e-2
    state_dependent_std: bool = False
    const_std: bool = True
    gc_encoder: nn.Module = None    
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(self, observations, goals=None, temperature=1.0, goal_encoded=True):
        """
        Return the action distribution.
        Args:
            observations: Observations.
            goals: Goals (optional).
            temperature: Scaling factor for the standard deviation.
            goal_encoded: Whether the goals are already encoded.
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)
        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))              
        return distribution