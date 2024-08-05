import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray as Key, Array
from typing import Tuple, Any


from optax import GradientTransformation



SGLDState = Tuple[Key, Any]



class SGLD:

    def __init__(self, epsilon, gamma, nbeta, seed):
        self.seed = seed
        self.epsilon = epsilon
        self.gamma = gamma
        self.nbeta = nbeta

        self.sqrt_epsilon = jnp.sqrt(self.epsilon)
        self.epsilon_nbeta = epsilon * nbeta / 2
        self.epsilon_gamma = epsilon * gamma / 2

    def init(self, params) -> SGLDState:
        return jax.random.key(self.seed), params

    def update(self, grads, state: SGLDState, params) -> Tuple[Any, SGLDState]:
        
        def single_param_update(key: Key, param: Array, init_param: Array, grad: Array) -> Array:
            print(f"param: {param.shape}")
            noise = self.sqrt_epsilon * jax.random.normal(key, shape=param.shape)
            prior_force = self.epsilon_gamma * (param - init_param)
            grad_force = self.epsilon_nbeta * grad

            

            print(f"\tNoise: min={noise.min()}, max={noise.max()}")
            print(f"\tPrior: min={prior_force.min()}, max={prior_force.max()}")
            print(f"\tGrad: min={grad_force.min()}, max={grad_force.max()}")

            

            return -prior_force - grad_force + noise
        
        key, init_params = state

        params_treedef = jax.tree.structure(params)
        keys = jax.random.split(key, num=params_treedef.num_leaves + 1)
        
        
        keys_updates = jax.tree.unflatten(params_treedef, keys[1:])
        key_state = keys[0]

        updates = jax.tree_map(single_param_update, keys_updates, params, init_params, grads)
        

        updates_flat, _ = jax.tree.flatten(updates)
        


        return updates, (key_state, init_params)

    
    
 



        










        
