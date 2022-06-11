from flax import linen as nn
import jax
import jax.numpy as jnp
import optax


class load(nn.Module):
    features: list

    @nn.compact
    def __call__(self, X, deterministic, rate = 0.03):
        for i, feature in enumerate(self.features):
            X = nn.Dense(feature, name=f"layer{i}")(X)
            if i != 0 and i != len(self.features) - 1:
                X = nn.relu(X)
                X = nn.Dropout(rate=rate, deterministic=deterministic)(X)
        return X

    def loss_fn(self, params, X, y, deterministic=False, key = jax.random.PRNGKey(0)):
        y_pred = self.apply(params, X, False, rngs={"dropout": key})
        loss = jnp.sum((y - y_pred)**2)/(2*X.shape[0])
        return loss


