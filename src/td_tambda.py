# Import required libraries
import numpy as np
from tqdm import tqdm
from src.decay_scheduler import decay_schedule


def td_lambda(policy, environment, discount_factor=1.0, initial_learning_rate=0.5,
              minimum_learning_rate=0.01, learning_rate_decay_ratio=0.3, lambda_factor=0.3,
              number_of_episodes=500):
    """
    Perform TD(lambda) algorithm.
    
    Parameters:
        policy (function): Policy function to follow
        environment (object): OpenAI Gym-like environment object
        discount_factor (float): Discount factor for future rewards, default 1.0
        initial_learning_rate (float): Initial learning rate, default 0.5
        minimum_learning_rate (float): Minimum learning rate, default 0.01
        learning_rate_decay_ratio (float): Ratio to decay learning rate, default 0.3
        lambda_factor (float): Eligibility trace decay factor, default 0.3
        number_of_episodes (int): Total number of episodes to simulate, default 500
    
    Returns:
        V (numpy.ndarray): Final value function
        V_track (numpy.ndarray): Evolution of value function over episodes
    """
    # Initialize the state-value function and the eligibility traces
    number_of_states = environment.observation_space.n
    state_value_function = np.zeros(number_of_states)
    value_function_track = np.zeros((number_of_episodes, number_of_states))
    eligibility_traces = np.zeros(number_of_states)

    # Create a learning rate decay schedule
    learning_rates = decay_schedule(initial_learning_rate, minimum_learning_rate,
                                    learning_rate_decay_ratio, number_of_episodes)

    # Loop over each episode
    for episode in tqdm(range(number_of_episodes), leave=False):
        # Reset the eligibility traces
        eligibility_traces.fill(0)
        # Start a new episode and get the initial state
        current_state, done = environment.reset(), False
        # Continue looping until the episode is done
        while not done:
            # Get an action from the policy
            action = policy(current_state)
            # Take a step in the environment with the action
            next_state, reward, done, _ = environment.step(action)
            # Calculate the Temporal Difference target
            td_target = reward + discount_factor * state_value_function[next_state] * (not done)
            # Calculate the Temporal Difference error
            td_error = td_target - state_value_function[current_state]
            # Increase the eligibility of the current state
            eligibility_traces[current_state] += 1
            # Update the state-value function using the TD error and the eligibility traces
            state_value_function += learning_rates[episode] * td_error * eligibility_traces
            # Decay the eligibility traces
            eligibility_traces *= discount_factor * lambda_factor
            # Update the current state
            current_state = next_state
        # Track the evolution of the state-value function
        value_function_track[episode] = state_value_function

    return state_value_function, value_function_track
