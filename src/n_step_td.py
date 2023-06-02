import numpy as np
from src.decay_scheduler import decay_schedule
from tqdm import tqdm


def n_step_td(policy, environment, discount_factor=1.0, initial_learning_rate=0.5, min_learning_rate=0.01,
              learning_rate_decay_ratio=0.5, n_step=3, num_episodes=500):
    """
    An implementation of n-step Temporal Difference (TD) learning algorithm.
    It is used for estimating the value function of a policy in a given environment.

    Parameters:
    policy (callable): A function that takes a state and returns an action.
    environment (gym.Env): The environment to train the agent on.
    discount_factor (float): The discount factor for future rewards.
    initial_learning_rate (float): The initial learning rate.
    min_learning_rate (float): The minimum learning rate.
    learning_rate_decay_ratio (float): The ratio at which learning rate decays each episode.
    n_step (int): The number of steps for the n-step TD learning.
    num_episodes (int): The number of episodes for training.

    Returns:
    np.array, np.array: Estimated value function, value function for each episode.
    """

    # Number of states in the environment
    num_states = environment.observation_space.n

    # Initialize value function and value function tracker
    value_function = np.zeros(num_states)
    value_function_tracker = np.zeros((num_episodes, num_states))

    # Schedule for learning rate decay
    learning_rates = decay_schedule(initial_learning_rate, min_learning_rate, learning_rate_decay_ratio, num_episodes)

    # Discount factors for each step in n-step TD
    discount_factors = np.logspace(0, n_step + 1, num=n_step + 1, base=discount_factor, endpoint=False)

    # Loop through each episode
    for episode in tqdm(range(num_episodes), leave=False):
        state, terminated, truncated, trajectory = environment.reset(), False, False, []
        state = state[0]

        # Episode loop
        while True:
            trajectory = trajectory[1:]

            # Collect experiences until we reach n_step or the episode ends
            while not terminated and len(trajectory) < n_step:
                action = policy(state)
                next_state, reward, terminated, truncated, _ = environment.step(action)
                experience = {
                    'state': state,
                    'reward': reward,
                    'next_state': next_state,
                    'terminated': terminated,
                    'truncated': truncated
                }
                trajectory.append(experience)
                state = next_state
                if terminated or truncated:
                    break

            # Prepare variables for TD update
            num_experiences = len(trajectory)
            estimated_state = trajectory[0]['state']
            rewards = np.array([experience['reward'] for experience in trajectory])
            partial_return = discount_factors[:num_experiences] * rewards
            bootstrap_value = 0
            if not terminated:
                next_state = trajectory[-1]['next_state']
                bootstrap_value = discount_factors[-1] * value_function[next_state]

            # n-step TD target and error
            n_step_td_target = np.sum(np.append(partial_return, bootstrap_value))
            print(n_step_td_target, value_function, estimated_state)

            n_step_td_error = n_step_td_target - value_function[estimated_state]

            # Update value function
            value_function[estimated_state] = value_function[estimated_state] + learning_rates[episode] * n_step_td_error

            # Check if trajectory ends
            if len(trajectory) == 1 and (terminated or truncated):
                break

        # Track value function
        value_function_tracker[episode] = value_function

    return value_function, value_function_tracker
