import jax.numpy as jnp
import numpy as np
import jax


def data_sin(points=20, xrange=(-3, 3), std=0.3):
    xx = jnp.linspace(-1, 1, 1000).reshape(-1, 1)
    key = jax.random.PRNGKey(0)
    epsilons = jax.random.normal(key, shape=(3,)) * 0.02
    y_true = jnp.array([[x + 0.3 * jnp.sin(2 * jnp.pi * (x + epsilons[0])) +
                       0.3 * jnp.sin(4 * jnp.pi * (x + epsilons[1])) + epsilons[2]] for x in xx])
    yy = jnp.array([[x + 0.3 * jnp.sin(2 * jnp.pi * (x + epsilons[0])) + 0.3 * jnp.sin(4 *
                   jnp.pi * (x + epsilons[1])) + epsilons[2] + x*np.random.normal(0, std)] for x in xx])
    return xx.reshape(1000, 1), yy.reshape(1000, 1), y_true.reshape(1000, 1)


def data_hetero(points=20, xrange=(-3, 3), std=0.3):
    xx = jnp.linspace(-1, 1, 1000).reshape(-1, 1)
    key = jax.random.PRNGKey(0)
    y_true = jnp.array([[x*10*x] for x in xx])
    yy = jnp.array([[x*10*x + x*x*np.random.normal(0, std)] for x in xx])
    return xx.reshape(1000, 1), yy.reshape(1000, 1), y_true.reshape(1000, 1)
