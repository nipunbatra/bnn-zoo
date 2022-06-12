from typing import Sequence
import flax
import flax.linen as nn


class MLP(nn.Module):
    architecture: Sequence[int]
    @nn.compact
    def __call__(self, x):
        for i, n_neurons in enumerate(self.architecture):
          x = nn.Dense(n_neurons, name=f'layers_{i}')(x)
          x = nn.relu(x)
        x = nn.Dense(1, name=f'layers_{len(self.architecture)}')(x)
        return x