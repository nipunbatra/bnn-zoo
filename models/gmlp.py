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
        X = X.at[1].set(nn.activation.softplus(X[1]))
        return X

    def loss_fn(self, params, X, y):
        def loss(params, X, y):
            mean, var = self.apply(params, X)
            d = tfp.distributions.Normal(loc=mean, scale=var)
            return -d.log_prob(y)
        return jnp.mean(jax.vmap(loss, in_axes=(None, 0, 0))(params, X, y))
