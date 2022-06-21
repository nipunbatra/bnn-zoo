
from tkinter.messagebox import NO
import jax
import jax.numpy as jnp 
import matplotlib.pyplot as plt
def target_toy(key, x):
    epsilons = jax.random.normal(key, shape=(3,)) * 0.02
    return (
        x + 0.3 * jnp.sin(2 * jnp.pi * (x + epsilons[0])) + 0.3 * jnp.sin(4 * jnp.pi * (x + epsilons[1])) + epsilons[2]
    )
def load_data(n_points=100):
   
   key, subkey = jax.random.split(jax.random.PRNGKey(0))
   x = jax.random.uniform(key, shape=(n_points, 1), minval=0.0, maxval=0.5)
   x_test_1 = jnp.linspace(-0.5,0,100).reshape(100,1)
   x_test_2 = jnp.linspace(0.5,1,100).reshape(100,1)
   target_vmap = jax.vmap(target_toy, in_axes=(0, 0), out_axes=0)
   keys = jax.random.split(subkey, x.shape[0])
   y = target_vmap(keys,x)
   y_test_1 = target_vmap(keys,x_test_1)
   y_test_2 = target_vmap(keys,x_test_2)
#    plt.scatter(x_test_1,y_test_1)
   return x,y,x_test_1,y_test_1,x_test_2,y_test_2

