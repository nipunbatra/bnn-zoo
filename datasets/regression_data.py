import jax.numpy as jnp
import numpy as np
import jax
import jax.numpy as jnp 
def target_toy(key, x):
    epsilons = jax.random.normal(key, shape=(1,))*2
    return (
       10*x*x +x*x*epsilons
    )
def data_set(n_points=100):
    key, subkey = jax.random.split(jax.random.PRNGKey(4))
    x =jnp.linspace(-3,3 ,n_points).reshape(-1, 1)
    x_test_1 = jnp.linspace(-4,-3,n_points).reshape(-1, 1) 
    x_test_2 = jnp.linspace(3,4,n_points).reshape(-1, 1) 
    target_vmap = jax.vmap(target_toy, in_axes=(0, 0), out_axes=0)
    keys = jax.random.split(subkey, x.shape[0])
    y = target_vmap(keys,x)
    y_test_1 = target_vmap(keys,x_test_1)
    y_test_2 = target_vmap(keys,x_test_2)
    x_test = jnp.concatenate([x_test_1,x_test_2], axis=0)
    y_test = jnp.concatenate([y_test_1,y_test_2], axis=0)
    return x, y, x_test, y_test

    # if(poly):
    #   y_true = jnp.array([[x*10*x ] for x in xx])
    #   if(hetero):
    #     yy = jnp.array([[x*10*x +x*x*np.random.normal(0, std)] for x in xx])
    #     y_test_1 =  jnp.array([[x*10*x +x*x*np.random.normal(0, std)] for x in x_test_1])
    #     y_test_2 =  jnp.array([[x*10*x +x*x*np.random.normal(0, std)] for x in x_test_2])
    #   else:
    #     yy = jnp.array([[x*10*x + np.random.normal(0, std)] for x in xx])
    #     y_test_1 =  jnp.array([[x*10*x +np.random.normal(0, std)] for x in x_test_1])
    #     y_test_2 =  jnp.array([[x*10*x +np.random.normal(0, std)] for x in x_test_2])
    # else:
    #   y_true = jnp.array([[3*x ] for x in xx])
    #   if(hetero):
    #     yy = jnp.array([[x*3 + x*np.random.normal(0, std)] for x in xx])
    #     y_test_1 =  jnp.array([[3*x +x*np.random.normal(0, std)] for x in x_test_1])
    #     y_test_2 =  jnp.array([[3*x +x*np.random.normal(0, std)] for x in x_test_2])
    #   else:
    #     yy = jnp.array([[x*3 + np.random.normal(0, std)] for x in xx])
    #     y_test_1 =  jnp.array([[3*x +np.random.normal(0, std)] for x in x_test_1])
    #     y_test_2 =  jnp.array([[3*x +np.random.normal(0, std)] for x in x_test_2])
    # return xx.reshape(n_points,1), yy.reshape(n_points,1),x_test_1,y_test_1.reshape(int(n_points),1),x_test_2,y_test_2.reshape(int(n_points),1)

