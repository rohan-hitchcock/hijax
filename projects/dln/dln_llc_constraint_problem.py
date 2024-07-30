""" We leave jax here to solve a constraint problem which is part of computing 
    the LLC of a DLN. See Theorem 1 in Aoyagi (2024) "Consideration on the 
    learning efficiency of multiple-layered neural networks with linear units". 
    The constraints are given in Definition 3. Cf. Thereom B.1 in Furman and Lau 
    (2024) "Estimating the Local Learning Coefficient at Scale".

    In the notation of Furman and Lau, this function finds the set of indices 
    \Sigma satisfying the constraints outlined in Theorem B.1 and returns 
    \Delta_\Sigma := \{\Delta_s : s \in \Sigma\}, where 
    \Delta = \{\Delta_0, ... \Delta_M\}. This set is the component of the 
    constraint satisfaction problem which is actually required to compute the 
    LLC of a DLN. 
"""
from typing import List

from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import FEASIBLE, OPTIMAL

def compute_delta_sigma(delta: List[int]) -> List[int]:

    model = cp_model.CpModel()

    # represents \Sigma as a boolean array the same length as delta. 
    # sigma[i] == 1 indicates delta[i] should be present in the set
    sigma = [model.new_int_var(0, 1, f's{i}') for i in range(len(delta))]

    # We need a sufficiently large number to represent the minimum of an empty 
    # set (the solver cannot handle math.inf), recalling that the minimum of an 
    # empty set is +\infty. From constraints (1) and (3) we see that this is sufficient
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
        raise RuntimeError(f"Constraints to compute delta_sigma could not be satisfied ({status=}).")

    return delta_sigma
