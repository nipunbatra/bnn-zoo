import tensorflow_probability.substrates.jax as tfp
import jax.numpy as jnp
import jax
dist = tfp.distributions
def loss(mean,sigma,y):
    """
    mean : (n_samples,1) or (n_sample,) prediction mean 
    sigma : (n_samples,1) or (n_sample,) prediction sigma 
    y : (n_samples,1) or (n_sample,) Y co-ordinate of ground truth 
    """
    def loss_fn(mean, sigma, y):
        d = dist.Normal(loc=mean, scale=sigma)
        return -d.log_prob(y)
    return jnp.mean(jax.vmap(loss_fn, in_axes=(0, 0, 0))(mean, sigma, y))

def ace(dataframe):
    """
    dataframe : pandas dataframe with Ideal and Counts as column for regression calibration
    It can be directly used as 2nd output from calibration_regression in plot.py 
    """
    return(jnp.sum(jnp.abs(dataframe['Ideal'].values-dataframe['Counts'].values)))