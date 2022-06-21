import jax
import jax.numpy as jnp
import optax
from functools import partial

def fit(model, params, auxs, learning_rate = 0.01, epochs=1000, random_seed = 0, verbose=False):

    rng = jax.random.PRNGKey(random_seed)

    opt = optax.adam(learning_rate=learning_rate)
    opt_state = opt.init(params)

    loss_fn = partial(model.loss_fn, **auxs)
    loss_grad_fn = jax.value_and_grad(loss_fn)
    losses = []

    @jax.jit
    def one_epoch(params, opt_state, rng):
        loss_val, grads = loss_grad_fn(params, rng=rng)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    try:
        for i in range(epochs):
            rng, _ = jax.random.split(rng)
            params, opt_state, loss_val = one_epoch(params, opt_state, rng)
            losses.append(loss_val)
            if verbose and i % (epochs / 10) == 0:
                print('Loss step {}: '.format(i), loss_val)
    except KeyboardInterrupt:
        print(f"Completed {i} epochs.")
        return params, jnp.array(losses)    

    return params, jnp.array(losses)