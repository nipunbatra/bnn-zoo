from flax import linen as nn
import jax
import jax.numpy as jnp
import optax


class mlp(nn.Module):
    features: list
    activations: list
    dropout_rate: list

    @nn.compact
    def __call__(self, X, deterministic):
        if len(self.activations) != len(self.features) - 1:
            raise Exception(f"Length of activations should be equal to {len(self.features) - 1}")

        if len(self.dropout_rate) != len(self.features) - 1:
            raise Exception(f"Length of dropout_rate should be equal to {len(self.features) - 1}")

        for i, feature in enumerate(self.features):
            X = nn.Dense(feature,  kernel_init=jax.nn.initializers.glorot_normal(),name=f"{i}_Dense")(X)
            if i != len(self.features) - 1:
                X = self.activations[i](X)
                X = nn.Dropout(rate=self.dropout_rate[i], deterministic=deterministic, 
                                name=f"{i}_Dropout_{self.dropout_rate[i]}")(X)
            else:
                X = nn.sigmoid(X)
        return X
        
    def loss_fn(self, params, X, y, deterministic=True, rng = jax.random.PRNGKey(0)):
        y_pred = self.apply(params, X, deterministic = deterministic, rngs={"dropout": rng})
        # y_pred = nn.sigmoid(y_pred)
        def binary_cross_entropy(y_hat, y):
            # y_hat = nn.sigmoid(y_hat)
            bce = y * jnp.log(y_hat) + (1 - y) * jnp.log(1 - y_hat)
            return jnp.mean(-bce)
        return jnp.mean(jax.vmap(binary_cross_entropy, in_axes=(0, 0))(y_pred, y))

