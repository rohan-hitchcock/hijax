from typing import List
from functools import partial
import itertools

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray as Key, Int


# just for testing
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from shared.utils import register_model

@register_model(weights=['layers'])
class DeepLinearNetwork:

    layers: List[Array]

    def __post_init__(self) -> None:
        
        assert all(len(layer.shape) == 2 for layer in self.layers)
        assert all(
            layer_in.shape[1] == layer_out.shape[0] 
            for layer_in, layer_out in zip(self.layers, self.layers[1:])
        )

    def __call__(self, x: Array) -> Array:
        # x has shape [B, N] where N is the input dimension and B is the batch 
        # size, so our vectors are row vectors. jnp.linalg.multi_dot figures out 
        # the optimal order to several successive matrix multiplications
        return jnp.linalg.multi_dot([x, *self.layers])

    def loss(self, x: Array, y: Array) -> Array:
        y_pred = self(x)
        return jnp.mean((y - y_pred) ** 2)

    @jax.jit
    def rank(self) -> Int[Array, '1']:
        return jnp.linalg.matrix_rank(jnp.linalg.multi_dot(self.layers))

    @property 
    def layer_sizes(self) -> Int[Array, '{len(self.layers) + 1}']:
        sizes = [layer.shape[0] for layer in self.layers]
        return jnp.array(sizes + [self.layers[-1].shape[1]])

    @classmethod
    def initialize(cls, key: Key, layer_sizes: List[int]):

        assert len(layer_sizes) >= 3, "Must have at least 2 layers"

        shapes = zip(layer_sizes, layer_sizes[1:])
        key_init_layers = jax.random.split(key, num=len(layer_sizes) - 1)

        # TODO check this
        layers = [jax.random.normal(k, shape) for k, shape in zip(key_init_layers, shapes)]

        return cls(layers=layers)
    
    @classmethod
    def initialize_true(
        cls, 
        key: Key, 
        min_layers: int, 
        max_layers: int, 
        min_layer_size: int, 
        max_layer_size: int, 
        reduce_layer_rank_prob: float = 0.5
    ):

        assert min_layers >= 2, "Must have at least two layers"
        assert min_layer_size >= 1

        key, key_num_layers = jax.random.split(key)
        num_layers = jax.random.randint(key_num_layers, shape=(1,), minval=min_layers, maxval=max_layers).item()

        key, key_layer_sizes = jax.random.split(key)
        layer_sizes = jax.random.randint(key_layer_sizes, shape=(num_layers + 1,), minval=min_layer_size, maxval=max_layer_size)

        key, key_init_layers = jax.random.split(key)
        model = cls.initialize(key_init_layers, layer_sizes)


        for i, layer in enumerate(model.layers):

            key, key_choose_layer = jax.random.split(key)
            key, key_choose_rank = jax.random.split(key)

            if jax.random.uniform(key_choose_layer) > reduce_layer_rank_prob:
                continue
            
            smallest_dim = 0 if layer.shape[0] <= layer.shape[1] else 1
            max_rank = layer.shape[smallest_dim]


            rank = jax.random.randint(key_choose_rank, shape=(1,), minval=0, maxval=max_rank).item()

            if smallest_dim == 0:
                model.layers[i] = layer.at[rank:, :].set(0.0)

            else:
                model.layers[i] = layer.at[:, rank:].set(0.0)

        return model




def theoretical_llc(true_model: DeepLinearNetwork):

    rank = true_model.rank()
    layer_sizes = true_model.layer_sizes

    delta = layer_sizes - rank

    delta_sigma = _compute_delta_sigma(delta)
    delta_sigma_sum = sum(delta_sigma)
    ell = len(delta_sigma) - 1

    a = delta_sigma_sum - (jnp.ceil(delta_sigma_sum / ell) - 1) * ell
    
    llc = (
        0.5 * (rank ** 2 + rank * (layer_sizes[0] + layer_sizes[-1]))
        + a * (ell - a) / (4 * ell) 
        - ell * (ell - 1) / (4 * ell ** 2) * (delta_sigma_sum ** 2)
        + 0.5 * sum(d1 * d2 for d1, d2 in itertools.combinations(delta_sigma, r=2))
    )
    return llc


from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import IntVar, FEASIBLE, UNKNOWN, INFEASIBLE, OPTIMAL, MODEL_INVALID

def _compute_delta_sigma(delta):

    if hasattr(delta, 'tolist'): # handle jax and numpy arrays
        delta = delta.tolist()

    model = cp_model.CpModel()

    # represents \Sigma as a boolean array the same length as delta. 
    # sigma[i] == 1 indicates delta[i] should be present in the set
    sigma = [model.new_int_var(0, 1, f's{i}') for i in range(len(delta))]

    # We need a sufficiently large number to be chosen to represent the minimum 
    # of an empty set (the solver cannot handle math.inf), recalling that the 
    # minimum of an empty set is +\infty. From constraints (1) and (3) we see 
    # that this is sufficient
    big_number = sum(delta) + 1

    # represents {\Delta_s : s \in \Sigma} plus zeros for any s \notin \Sigma
    # these additional zeros OK as we only sum or take the maximum of this list
    delta_from_sigma = [d * s for d, s in zip(delta, sigma)] 
    
    # this variable is linear in sigma and so strictly should not be required, 
    # but the solver struggles if we do not add this intermediate variable
    sum_delta_from_sigma = model.new_int_var(0, sum(delta), 'sum_delta_from_sigma')
    model.add(sum_delta_from_sigma == sum(delta_from_sigma))

    # likewise, the solver struggles without this intermediate variable
    # inspection of constraints tells us ell >= 1
    ell = model.new_int_var(1, len(delta) - 1, 'ell')
    model.add(ell == sum(sigma) - 1)

    # intermediate variables for quantities which are not linear in sigma -----

    max_delta = max(delta) # for variable domains 
    
    # \max \{\Delta_s : s \in \Sigma\}
    max_delta_from_sigma = model.new_int_var(0, max_delta, 'max_delta_from_sigma')
    model.add_max_equality(max_delta_from_sigma, delta_from_sigma)

    # \min \{\Delta_s : s \notin \Sigma\}   
    # To compute the minimum, we put d (from delta) into the set if the 
    # corresponding s (from sigma) == 0 otherwise, we put a sufficently large 
    # value into the set that will only be chosen as the minimum if sigma = [1, 1, ... ]. 
    min_delta_not_from_sigma = model.new_int_var(0, big_number, 'min_delta_not_from_sigma')    
    model.add_min_equality(min_delta_not_from_sigma, [d * (1 - s) + big_number * s for d, s in zip(delta, sigma)] )

    # ell * \max \{\Delta_s : s \in \Sigma\}
    ell_times_max_delta_from_sigma = model.new_int_var(0, (len(delta) - 1) * max_delta, 'ell_times_max_delta_from_sigma')
    model.add_multiplication_equality(ell_times_max_delta_from_sigma, [ell, max_delta_from_sigma])

    # ell * \min \{\Delta_s : s \notin \Sigma\}
    ell_times_min_delta_not_from_sigma = model.new_int_var(0, (len(delta) - 1) * big_number, 'ell_times_min_delta_not_from_sigma')
    model.add_multiplication_equality(ell_times_min_delta_not_from_sigma, [ell, min_delta_not_from_sigma])

    # Constraints from theorem ------------------------------------------------

    # See Aoyagi (2024) Definition 3 (cf. Furman and Lau (2024) Theorem B.1)
    # When \Sigma ^c = \emptyset we have arranged things so that 
    # min_delta_not_from_sigma = sum(delta) + 1, which is sufficiently large 
    # to satisfy constraints (1) and (3). When \Sigma = \emptyset the last 
    # constraint will always be violated (we also disallow this with the 
    # domain of ell)
    model.add(max_delta_from_sigma < min_delta_not_from_sigma)
    model.add(sum_delta_from_sigma >= ell_times_max_delta_from_sigma)
    model.add(sum_delta_from_sigma < ell_times_min_delta_not_from_sigma)

    # Solve the system --------------------------------------------------------
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == FEASIBLE or status == OPTIMAL:
        delta_sigma = [delta[i] for i, s in enumerate(sigma) if solver.value(s) == 1]
    else:
        raise RuntimeError("Constraints to compute delta_sigma could not be satisfied.")

    return delta_sigma


def _debug_compute_delta_sigma(delta, sigma):

    cp_model = _compute_delta_sigma(delta)

    
    sum_delta_from_sigma = sum(d * s for d, s in zip(delta, sigma))
    ell = sum(sigma) - 1
    max_delta_from_sigma = max(d * s for d, s in zip(delta, sigma))
    min_delta_not_from_sigma = min(d * (1 - s) for d, s in zip(delta, sigma))
    ell_times_max_delta_from_sigma = ell * max_delta_from_sigma
    ell_times_min_delta_not_from_sigma = ell * min_delta_not_from_sigma
    is_sigma_full = (ell == len(delta) - 1)

    print("Model variables:")
    for var in cp_model.Proto().variables:
        print(var)
    

    print("Actual variables:")
    print(f"{sum_delta_from_sigma=}")
    print(f"{ell=}")
    print(f"{max_delta_from_sigma=}")
    print(f"{min_delta_not_from_sigma=}")
    print(f"{ell_times_max_delta_from_sigma=}")
    print(f"{ell_times_min_delta_not_from_sigma=}")
    print(f"{is_sigma_full=}")


    print("Actual constraints:")

    print(f"(1): {is_sigma_full or (max_delta_from_sigma < min_delta_not_from_sigma)}")
    print(f"(2): {sum_delta_from_sigma >= ell_times_max_delta_from_sigma}")
    print(f"(3): {is_sigma_full or (sum_delta_from_sigma < ell_times_min_delta_not_from_sigma)}")
   

if __name__ == "__main__":

    key = jax.random.key(2)
    model_true = DeepLinearNetwork.initialize_true(key, min_layers=2, max_layers=3, min_layer_size=4, max_layer_size=10)


    import time
    from statistics import mean

    import tqdm
    print(f"{FEASIBLE=}, {UNKNOWN=}, {INFEASIBLE=}, {OPTIMAL=}, {MODEL_INVALID=}")
    print()


    sigma = [0, 1, 1]
    delta = [2, 1, 0]
    """print([d * (1 - s) for d, s in zip(delta, sigma)])

    _debug_compute_delta_sigma([2, 1, 0], [0, 1, 1])


    exit()"""

    times = []
    for _ in tqdm.trange(20):

        key, key_test = jax.random.split(key)
        model_true = DeepLinearNetwork.initialize_true(key, min_layers=10, max_layers=15, min_layer_size=5, max_layer_size=50)

    
        rank = model_true.rank()
        delta = model_true.layer_sizes - model_true.rank()


        print("Start solving constraints")
        start = time.time()
        _compute_delta_sigma(delta)
        end = time.time()
        times.append(end - start)
        

    print(f"{mean(times)=}")






    # llc = theoretical_llc(model_true)

    # print(llc)
    


    