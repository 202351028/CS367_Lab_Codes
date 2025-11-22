import numpy as np
import matplotlib.pyplot as plt
from policy_evaluation_gbike import policy_evaluation
from policy_improvement_gbike import policy_improvement_gbike

lambda_rent = np.array([3.0, 4.0])  # expected rental requests at loc1 and loc2
lambda_ret  = np.array([3.0, 2.0])  # expected returns at loc1 and loc2

params = {
    'max_bikes': 20,
    'max_move': 5,
    'rent_reward': 10,
    'move_cost': 2,
    'free_transfer': 1,
    'parking_penalty': 4,
    'poisson_upper': 11,   # use 0..10 with tail folded
    'gamma': 0.9
}

# --------------------
# Policy iteration driver
# --------------------

def policy_iteration(max_iters=50):
    max_bikes = params['max_bikes']
    policy = np.zeros((max_bikes+1, max_bikes+1), dtype=int)  # start with zero-action policy

    for it in range(1, max_iters+1):
        print(f"Policy Iteration: evaluation step (iteration {it}) ...")
        V = policy_evaluation(policy, lambda_rent, lambda_ret, params, theta=1e-3)

        print("Policy improvement ...")
        new_policy = policy_improvement_gbike(V, lambda_rent, lambda_ret, params)

        if np.array_equal(policy, new_policy):
            print(f"Policy converged after {it} iterations.")
            return new_policy, V
        policy = new_policy

    print("Policy iteration reached max iterations.")
    return policy, V

if __name__ == "__main__":
    policy, V = policy_iteration()

    np.set_printoptions(precision=2, suppress=True)
    print("Final policy (net moved from loc1 -> loc2):")
    print(policy)

    print("\nValue function sample (V shape = (s1,s2)):")
    print(V)

    # plot heatmap for V and textual policy
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im = axes[0].imshow(V, origin='lower', cmap='viridis')
    axes[0].set_title('Value function V(s1,s2)')
    axes[0].set_xlabel('s2 (loc2 bikes)')
    axes[0].set_ylabel('s1 (loc1 bikes)')
    fig.colorbar(im, ax=axes[0])

    # policy text grid
    arrow_map = { -5:'←5', -4:'←4', -3:'←3', -2:'←2', -1:'←1', 0:'·', 1:'1→', 2:'2→', 3:'3→', 4:'4→', 5:'5→' }
    policy_str = np.vectorize(lambda a: arrow_map.get(int(a), str(int(a))))(policy)
    axes[1].axis('off')
    axes[1].set_title('Policy (net 1->2)')
    table_text = "\n".join([" ".join(row) for row in policy_str])
    axes[1].text(0.01, 0.99, table_text, fontsize=8, va='top', family='monospace')

    plt.tight_layout()
    plt.show()
