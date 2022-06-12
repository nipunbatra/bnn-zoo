import jax.numpy as jnp
import numpy as np
def data_set(n_points=20, xrange=3, std=3.,poly=False,hetero=False):
    xx =jnp.linspace(-1*xrange,1*xrange ,n_points).reshape(-1, 1)
    x_test_1 = jnp.linspace(-1*(xrange+1),-1*xrange,int(n_points/2)).reshape(-1, 1) 
    x_test_2 = jnp.linspace(xrange,xrange+1,int(n_points/2)).reshape(-1, 1) 
    if(poly):
      y_true = jnp.array([[x*10*x ] for x in xx])
      if(hetero):
        yy = jnp.array([[x*10*x +x*x*np.random.normal(0, std)] for x in xx])
        y_test_1 =  jnp.array([[x*10*x +x*x*np.random.normal(0, std)] for x in x_test_1])
        y_test_2 =  jnp.array([[x*10*x +x*x*np.random.normal(0, std)] for x in x_test_2])
      else:
        yy = jnp.array([[x*10*x + np.random.normal(0, std)] for x in xx])
        y_test_1 =  jnp.array([[x*10*x +np.random.normal(0, std)] for x in x_test_1])
        y_test_2 =  jnp.array([[x*10*x +np.random.normal(0, std)] for x in x_test_2])
    else:
      y_true = jnp.array([[3*x ] for x in xx])
      if(hetero):
        yy = jnp.array([[x*3 + x*np.random.normal(0, std)] for x in xx])
        y_test_1 =  jnp.array([[3*x +x*np.random.normal(0, std)] for x in x_test_1])
        y_test_2 =  jnp.array([[3*x +x*np.random.normal(0, std)] for x in x_test_2])
      else:
        yy = jnp.array([[x*3 + np.random.normal(0, std)] for x in xx])
        y_test_1 =  jnp.array([[3*x +np.random.normal(0, std)] for x in x_test_1])
        y_test_2 =  jnp.array([[3*x +np.random.normal(0, std)] for x in x_test_2])
    return xx.reshape(n_points,1), yy.reshape(n_points,1),y_true.reshape(n_points,1),x_test_1,y_test_1.reshape(int(n_points/2),1),x_test_2,y_test_2.reshape(int(n_points/2),1)

