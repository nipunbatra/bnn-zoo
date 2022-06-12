
import jaxopt
import optax
from typing import Sequence
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import flax.linen as nn

class GaussianMLP(nn.Module):
    architecture: Sequence[int]
    @nn.compact
    def __call__(self, x):
        for i, n_neurons in enumerate(self.architecture):
          x = nn.Dense(n_neurons, name=f'layers_{i}')(x)
          x = nn.relu(x)
        x = nn.Dense(2, name=f'layers_{len(self.architecture)}')(x)
        x = x.at[1].set(nn.activation.softplus(x[1]))
        
        return x
    


    

    