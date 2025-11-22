# policy_improvement_gbike.py
import numpy as np
from math import factorial, exp

def poisson_pmf_array(lmbda, upper):
    n = np.arange(upper)
    pmf = np.exp(-lmbda) * (lmbda ** n) / np.vectorize(factorial)(n)
    tail = 1.0 - pmf.sum()
    if tail > 0:
        pmf[-1] += tail
    return pmf

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def expected_transition_and_reward(s1_after_move, s2_after_move, P_rent_1, P_rent_2, P_ret_1, P_ret_2, max_bikes, reward_per_rental):
    exp_rental_reward = 0.0
    next_state_probs = {}
    for n1, p_n1 in enumerate(P_rent_1):
        if p_n1 == 0:
            continue
        fulfilled1 = min(n1, s1_after_move)
        for n2, p_n2 in enumerate(P_rent_2):
            if p_n2 == 0:
                continue
            fulfilled2 = min(n2, s2_after_move)
            prob_rent = p_n1 * p_n2
            exp_rental_reward += prob_rent * reward_per_rental * (fulfilled1 + fulfilled2)
            s1_after_rent = s1_after_move - fulfilled1
            s2_after_rent = s2_after_move - fulfilled2
            for r1, p_r1 in enumerate(P_ret_1):
                if p_r1 == 0:
                    continue
                for r2, p_r2 in enumerate(P_ret_2):
                    if p_r2 == 0:
                        continue
                    prob = prob_rent * p_r1 * p_r2
                    s1_next = clamp(s1_after_rent + r1, 0, max_bikes)
                    s2_next = clamp(s2_after_rent + r2, 0, max_bikes)
                    key = (s1_next, s2_next)
                    next_state_probs[key] = next_state_probs.get(key, 0.0) + prob
    return exp_rental_reward, list(next_state_probs.items())

def policy_improvement_gbike(V,lambda_rent, lambda_ret, params):

    max_bikes = params['max_bikes']
    max_move = params['max_move']
    rent_reward = params['rent_reward']
    move_cost_unit = params['move_cost']
    free_transfer = params['free_transfer']
    parking_penalty = params['parking_penalty']
    poisson_upper = params['poisson_upper']
    gamma = params['gamma']

    # Precompute Poisson pmfs
    P_rent_1 = poisson_pmf_array(lambda_rent[0], poisson_upper)
    P_rent_2 = poisson_pmf_array(lambda_rent[1], poisson_upper)
    P_ret_1  = poisson_pmf_array(lambda_ret[0], poisson_upper)
    P_ret_2  = poisson_pmf_array(lambda_ret[1], poisson_upper)

    policy = np.zeros((max_bikes+1, max_bikes+1), dtype=int)

    for s1 in range(max_bikes+1):
        for s2 in range(max_bikes+1):
            best_a = None
            best_val = -np.inf

            # feasible action bounds
            amin = max(-max_move, -s2, -(max_bikes - s1))
            amax = min(max_move, s1, max_bikes - s2)
            for a in range(amin, amax+1):
                s1_after_move = clamp(s1 - a, 0, max_bikes)
                s2_after_move = clamp(s2 + a, 0, max_bikes)

                if a > 0:
                    move_cost = move_cost_unit * max(0, a - free_transfer)
                else:
                    move_cost = move_cost_unit * abs(a)

                parking_cost = 0
                if s1_after_move > 10:
                    parking_cost += parking_penalty
                if s2_after_move > 10:
                    parking_cost += parking_penalty

                exp_rental_rev, next_state_probs = expected_transition_and_reward(
                    s1_after_move, s2_after_move,
                    P_rent_1, P_rent_2, P_ret_1, P_ret_2,
                    max_bikes, rent_reward)

                immediate_expected_reward = exp_rental_rev - move_cost - parking_cost

                exp_future_v = 0.0
                for (ns1, ns2), prob in next_state_probs:
                    exp_future_v += prob * V[ns1, ns2]

                qsa = immediate_expected_reward + gamma * exp_future_v

                if qsa > best_val:
                    best_val = qsa
                    best_a = a

            policy[s1, s2] = best_a if best_a is not None else 0

    return policy
