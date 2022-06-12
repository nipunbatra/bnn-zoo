import tensorflow_probability.substrates.jax as tfp
import jaxopt
import optax
import jax
import jax.numpy as jnp
def predict(model,params,X):
  y_pred  = model.apply(params,X)
  return y_pred
def fit_ensemble(n_models,model,X,y,verbose=True,learning_rate=1e-1,n_epochs=1000):
  means_list=[]
  for j in range(n_models):
    theta = fit(model,X,y,verbose,learning_rate,n_epochs,42+j)
    means_list.append(predict(model,theta,X))
  final_mean = jnp.stack(means_list).mean(axis=0)
  final_sigma = jnp.stack(means_list).std(axis=0)
  return final_mean,final_sigma
def fit(model,X,y,verbose=True,learning_rate=1e-1,n_epochs=1000,random_state=42):
  key1, key2 = jax.random.split(jax.random.PRNGKey(random_state))
  init_var= jax.random.normal(key1, (1,))
  params = model.init(key2, init_var) 
  def mse(params, x_batched, y_batched):
    def squared_error(x, y):
      pred = model.apply(params, x)
      return jnp.inner(y-pred, y-pred) / 2.0
    return jnp.mean(jax.vmap(squared_error)(x_batched,y_batched), axis=0)
  solver = jaxopt.OptaxSolver(opt=optax.adam(learning_rate), fun=mse, maxiter=1000, tol=1e-8)
  @jax.jit
  def jit_update(theta, state, data):
      X, y = data
      return solver.update(theta, state, X, y)
  state = solver.init_state(params)
  # loss = []
  for i in range(n_epochs):
    params, state = jit_update(params, state, (X, y))
    # loss.append(loss_batch(params, X, y))
    if verbose:
      if i%50==0:
        print(mse(params, X, y))
  return params