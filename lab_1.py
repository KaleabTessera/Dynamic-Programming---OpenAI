import numpy as np
from environments.gridworld import GridworldEnv
import timeit
import matplotlib.pyplot as plt


def policy_evaluation(env, policy, discount_factor=1.0, theta=0.00001):
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for state in range(env.observation_space.n):
            v = 0
            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, done in env.P[state][action]:
                    v += action_prob * prob * \
                        (reward + discount_factor * V[next_state])
            delta = max(delta, np.abs(v - V[state]))
            V[state] = v
        if delta < theta:
            break
    return np.array(V)


def policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=1.0):

    def one_step_lookahead(state, V):
        A = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    # Ini Policy -Random
    policy = np.zeros((env.observation_space.n, env.action_space.n))
    policy.fill(1/env.action_space.n)
    # Terminal State
    policy[-1] = np.zeros(env.action_space.n)
    policy_stable = True
    while(True):
        # Policy Evalutaion
        V = policy_evaluation_fn(env, policy, discount_factor=discount_factor)
        policy_stable = True
        # Policy Improvement - -1 ignore terminal state
        for s in range(env.observation_space.n):
            best_a_current_policy = np.argmax(policy[s])
            best_a_new_policy = np.argmax(one_step_lookahead(s, V))
            if(best_a_current_policy != best_a_new_policy):
                policy_stable = False
            # Greedy
            new_policy_s = np.zeros(env.action_space.n)
            new_policy_s[best_a_new_policy] = 1
            policy[s] = new_policy_s

        if(policy_stable):
            return policy, V


def value_iteration(env, theta=0.0001, discount_factor=1.0):

    def one_step_lookahead(state, V):
        A = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    V = np.zeros(env.observation_space.n)
    while(True):
        delta = 0
        for s in range(env.observation_space.n):
            v = V[s]
            # Q = np.zeros(env.action_space.n)
            best_action = np.max(one_step_lookahead(s, V))
            V[s] = best_action
            delta = max(delta, np.abs(v - V[s]))
        if(delta < theta):
            break
    policy = np.zeros((env.observation_space.n, env.action_space.n))

    # Policy that follows the optimal value
    for s in range(env.observation_space.n):
        best_action_index = np.argmax(one_step_lookahead(s, V))
        policy[s, best_action_index] = 1
    return policy, V


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


def main():
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(shape=[5, 5], terminal_states=[
                       24], terminal_reward=0, step_reward=-1)
    state = env.reset()
    print("")
    env.render()
    print("")

    # TODO: generate random policy
    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")

    directions = {
        0: "↑",
        1: "→",
        2: "↓",
        3: "←"
    }
    # Random Policy
    random_p = np.zeros((env.observation_space.n, env.action_space.n))
    random_p.fill(1/env.action_space.n)
    # Terminal State
    random_p[-1] = np.zeros(env.action_space.n)

    maps = np.chararray(env.observation_space.n, unicode=True)
    maps.fill('o')
    terminal_states = [24]
    for term in terminal_states:
        maps[term] = 'T'
    for t in range(100):
        action = env.action_space.sample()
        current_s = env.get_current_state()
        state, reward, done, info = env.step(action)
        maps[current_s] = directions[action]
        if(done):
            print("Done")
            break
    maps[state] = 'X'
    env.close()
    print(maps.reshape(5, 5))

    # # TODO: evaluate random policy
    v = policy_evaluation(env, policy=random_p)

    # # TODO: print state value for each state, as grid shape
    print(v)
    # # Test: Make sure the evaluated policy is what we expected
    expected_v = np.array([-106.81, -104.81, -101.37, -97.62, -95.07,
                           -104.81, -102.25, -97.69, -92.40, -88.52,
                           -101.37, -97.69, -90.74, -81.78, -74.10,
                           -97.62, -92.40, -81.78, -65.89, -47.99,
                           -95.07, -88.52, -74.10, -47.99, 0.0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")
    # TODO: use  policy improvement to compute optimal policy and state values
    # policy, v = [], []  # call policy_iteration
    policy, v = policy_iteration(
        env, policy_evaluation)  # call policy_iteration

    # TODO Print out best action for each state in grid shape
    best_a = np.chararray(env.observation_space.n, unicode=True)
    maps.fill('o')
    for s in range(env.observation_space.n):
        best_a[s] = directions[np.argmax(policy[s])]
    terminal_states = [24]
    for term in terminal_states:
        best_a[term] = 'T'
    print(best_a.reshape(5, 5))
    # TODO: print state value for each state, as grid shape
    print(v.reshape(5, 5))
    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")
    # TODO: use  value iteration to compute optimal policy and state values
    policy, v = value_iteration(env)  # call value_iteration

    # # TODO Print out best action for each state in grid shape

    # # TODO: print state value for each state, as grid shape

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    discounts = np.logspace(-0.2, 0, num=30)
    v = []
    p = []
    for d in discounts:
        wrapped_v = wrapper(value_iteration, env, discount_factor=d)
        value_v = timeit.timeit(wrapped_v, number=10)
        v.append(value_v)

        wrapped_p = wrapper(value_iteration, env, discount_factor=d)
        value_p = timeit.timeit(wrapped_p, number=10)
        p.append(value_p)

    plt.plot(discounts, v, label='Value-Iteration')
    plt.plot(discounts, p, label='Policy-Iteration')
    plt.ylabel('Average time over ten runs')
    plt.xlabel('$\gamma$')
    plt.title('Discount factor vs Average time')
    plt.legend()
    # Not to show the graph everytime
    # plt.show()


if __name__ == "__main__":
    main()
