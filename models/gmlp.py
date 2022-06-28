import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp


class gmlp(nn.Module):
    features: list

    @nn.compact
    def __call__(self, X):
        for i, feature in enumerate(self.features):
            X = nn.Dense(feature, name=f"layer{i}")(X)
            if i != len(self.features):
                X = nn.relu(X)
        X = nn.Dense(2, name=f'layers_{len(self.features)}')(X)
        mean = X[:, 0]
        sigma = nn.softplus(X[:, 1])
        return mean, sigma

    def loss_fn(self, params, X, y):
        mean, sigma = self.apply(params, X)

        @jax.jit
        def loss(params, mean, sigma, y):
            d = tfp.distributions.Normal(loc=mean, scale=sigma)
            return -d.log_prob(y)
        return jnp.mean(jax.vmap(loss, in_axes=(None, 0, 0, 0))(params, mean, sigma, y))
