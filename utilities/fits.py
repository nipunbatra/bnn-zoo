import jax
import jax.numpy as jnp
import optax

def fit(model, params, X, y, learning_rate = 0.01, epochs=1000, verbose=False):
    
    opt = optax.adam(learning_rate=learning_rate)
    opt_state = opt.init(params)

    loss_grad_fn = jax.jit(jax.value_and_grad(model.loss_fn))
    losses = []
    for i in range(epochs):
        loss_val, grads = loss_grad_fn(params, X, y)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        losses.append(loss_val)
        if verbose and i % (epochs / 10) == 0:
            print('Loss step {}: '.format(i), loss_val)
    return params, jnp.array(losses)