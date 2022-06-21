import jax.numpy as jnp
def predict(n_models,model,params_list,X):
    means_list = []
    sigmas_list = []
    for i in range(n_models):
        mean,sigma = model.apply(params_list[i], X)
        means_list.append(mean)
        sigmas_list.append(sigma)
    means = jnp.stack(means_list)
    final_mean = means.mean(axis=0)
    sigmas = jnp.stack(sigmas_list)
    final_sigma = (sigmas + means**2).mean(axis=0) - final_mean**2
    return final_mean,final_sigma


