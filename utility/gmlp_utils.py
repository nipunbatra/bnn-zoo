import tensorflow_probability.substrates.jax as tfp
import jaxopt
import optax
import jax
import jax.numpy as jnp

def fit_ensemble(n_models,model,X,y,verbose=True,learning_rate=1e-1,n_epochs=1500):
  means_list = []
  sigmas_list = []
  for j in range(n_models):
    theta = fit(model,X,y,verbose=verbose,random_state=j+42)
    mean,sigma = predict(model,theta,X)
    means_list.append(mean)
    sigmas_list.append(sigma)
  means = jnp.stack(means_list)
  final_mean = means.mean(axis=0)
  sigmas = jnp.stack(sigmas_list)
  final_sigma = (sigmas + means**2).mean(axis=0) - final_mean**2
  return final_mean,final_sigma
def predict(model,params,X):
  y_pred = model.apply(params,X)
  mean,sigma = y_pred[:,0],y_pred[:,1]
  return mean,sigma
def fit(model,X,y,verbose=True,learning_rate=1e-1,n_epochs=1500,random_state=42):
  key1, key2 = jax.random.split(jax.random.PRNGKey(random_state))
  init_var= jax.random.normal(key1, (1,))
  params = model.init(key2, init_var) 
  def gaussian_loss(params, x, y):
    dist = tfp.distributions
    mean, var = model.apply(params, x)
    d = dist.Normal(loc=mean, scale=var)
    return -d.log_prob(y)
  def loss_batch(params,x,y,func=gaussian_loss):
    return jnp.mean(jax.vmap(func, in_axes=(None, 0, 0))(params, x.reshape(-1, 1), y))
  solver = jaxopt.OptaxSolver(opt=optax.adam(learning_rate), fun=loss_batch, maxiter=1000, tol=1e-8)
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
        print(loss_batch(params, X, y))
  return params