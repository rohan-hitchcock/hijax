import jax
import jax.numpy as jnp

def gradient_descent_step(weights, grad_weights, learning_rate):
    update_weight = lambda w, grad_w : w - learning_rate * grad_w
    return jax.tree.map(update_weight, weights, grad_weights)


def sgld_step(key, weights, initial_weights, grad_weights, learning_rate, gamma, beta, noise_level=1.0):


    def update_weight(w, w_0, grad_w, key):

        noise = jnp.sqrt(learning_rate * noise_level) * jax.random.normal(key, shape=w.shape)

        dw = - (learning_rate / 2) * (gamma * (w - w_0) + beta * grad_w) + noise

        return w + dw
    
    # create a different key for each weight leaf and put into the same tree 
    # structure as weights
    weights_structure = jax.tree.structure(weights)
    keys = jax.random.split(key, num=weights_structure.num_leaves)
    keys = jax.tree.unflatten(weights_structure, keys)

    return jax.tree.map(update_weight, weights, initial_weights, grad_weights, keys)