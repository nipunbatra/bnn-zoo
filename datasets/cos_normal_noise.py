import jax
import jax.numpy as jnp


def load(min_x=-8, max_x=8, n_samples=100, random_state=0):

    key = jax.random.PRNGKey(random_state)

    fn = lambda x: 0.1 * x * jnp.cos(x)

    X = jnp.linspace(min_x, max_x, num=n_samples).reshape(-1, 1)
    x_test1 = jnp.linspace(min_x - 2, min_x, num=n_samples // 10).reshape(-1, 1)
    x_test2 = jnp.linspace(max_x, max_x + 2, num=n_samples // 10).reshape(-1, 1)

    y = fn(X) + 0.05 * jax.random.normal(key, shape=[n_samples, 1])
    y_test1 = fn(x_test1) + 0.05 * jax.random.normal(key, shape=[n_samples // 10, 1])
    y_test2 = fn(x_test2) + 0.05 * jax.random.normal(key, shape=[n_samples // 10, 1])

    return [X, x_test1, x_test2], [y, y_test1, y_test2]
