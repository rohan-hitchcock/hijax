import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray as Key, Array
from typing import Tuple, Any


SGLDState = Tuple[Key, Any]



class SGLD:

    def __init__(self, epsilon, gamma, nbeta):
        self.epsilon = epsilon
        self.gamma = gamma
        self.nbeta = nbeta

    def init(self, params, key: Key) -> SGLDState:
        return key, params

    def update(self, grads, state: SGLDState, params) -> Tuple[Any, SGLDState]:
        
        def single_param_update(key: Key, param: Array, init_param: Array, grad: Array) -> Array:
            
            noise = jnp.sqrt(self.epsilon) * jax.random.normal(key, shape=param.shape)
            
            return -0.5 * self.epsilon * self.gamma * (param - init_param) - 0.5 * self.epsilon * self.nbeta * grad + noise
        
        key, init_params = state
        key_next, key_curr = jax.random.split(key)


        params_treedef = jax.tree.structure(params)
        keys_param_updates = jax.random.split(key_curr, num=params_treedef.num_leaves)
        
        
        keys_updates = jax.tree.unflatten(params_treedef, keys_param_updates)
        updates = jax.tree_map(single_param_update, keys_updates, params, init_params, grads)

        return updates, (key_next, init_params)

    
    
 



        










        
