import logging  ## for ignoring check warnings

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

from ajax.advi import ADVI
from ajax.utils import train

import flax.linen as nn
from flax.core.frozen_dict import freeze, unfreeze

import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from functools import partial


class MLP(nn.Module):
    features: list
    activations: list

    @nn.compact
    def __call__(self, X):
        if len(self.activations) != len(self.features) - 1:
            raise Exception(
                f"Length of activations should be equal to {len(self.layers) - 1}"
            )

        for i, feature in enumerate(self.features):
            X = nn.Dense(feature, name=f"{i}_Dense")(X)
            if i != len(self.features) - 1:
                X = self.activations[i](X)
        return X.ravel()


def vi_model(mlp_features, x_train, y_train, n_epochs=50000, variable_noise=True):
    """
    function to return trained VI model in ajax
    mlp_features : features for MLP , [dimensions, activations]
    x_train : training data 
    y_train : training output (n_samples,)
    """

    mlp = MLP(*mlp_features)
    seed = jax.random.PRNGKey(89)
    frozen_params = mlp.init(seed, x_train)
    params = unfreeze(frozen_params)
    prior = jax.tree_map(
        lambda param: tfd.Independent(
            tfd.Normal(loc=jnp.zeros(param.shape), scale=jnp.ones(param.shape)),
            reinterpreted_batch_ndims=len(param.shape),
        ),
        params,
    )

    bijector = jax.tree_map(lambda param: tfb.Identity(), params)

    def get_log_likelihood(latent_sample, data, aux, **kwargs):
        frozen_params = freeze(latent_sample)
        y_pred = mlp.apply(frozen_params, aux["X"])
        scale = jnp.exp(kwargs["log_noise_scale"])
        if variable_noise == False:
            scale = 0.1
        return tfd.Normal(loc=y_pred, scale=scale).log_prob(data).sum()

    model = ADVI(prior, bijector, get_log_likelihood, vi_type="mean_field")

    params = model.init(jax.random.PRNGKey(8))
    mean = params["posterior"].mean()
    params["posterior"] = tfd.MultivariateNormalDiag(
        loc=mean,
        scale_diag=jax.random.normal(jax.random.PRNGKey(3), shape=(len(mean),)) - 10,
    )
    params["log_noise_scale"] = 0.001

    tx = optax.adam(learning_rate=0.001)
    seed1 = jax.random.PRNGKey(4)
    seed2 = jax.random.PRNGKey(5)

    loss_fn = partial(
        model.loss_fn,
        aux={"X": x_train},
        batch=y_train,
        data_size=len(y_train),
        n_samples=1,
    )
    results = train(
        loss_fn,
        params,
        n_epochs=n_epochs,
        optimizer=tx,
        seed=seed2,
        return_args={"losses"},
    )
    return mlp, model, results


def vi_predict(vi_model, results, mlp_model, data):
    """
    Function to predict given vi_model in ajax
    vi_model : VI model in ajax
    results : result obtained after training ajax VI model
    mlp_model : mlp model for which ajax VI model was trained
    data : data for which prediction is requires 
    """

    posterior = vi_model.apply(results["params"])
    seed = jax.random.PRNGKey(4)
    weights = posterior.sample(seed, sample_shape=(1000,))

    def draw_sample(weights):
        y_pred = mlp_model.apply(freeze(weights), data)
        return y_pred

    y_samples = jax.vmap(draw_sample)(weights)
    return y_samples
