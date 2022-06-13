import jax.numpy as jnp
import numpy as np
def data(points=20, xrange=(-3, -3), std=0.3):
    xx = jnp.linspace(-1, 1, 1000).reshape(-1, 1)
    y_true = jnp.array([[x*10*x] for x in xx])
    yy = jnp.array([[x*10*x + x*x*np.random.normal(0, std)] for x in xx])
    return xx.reshape(1000, 1), yy.reshape(1000, 1), y_true.reshape(1000, 1)
