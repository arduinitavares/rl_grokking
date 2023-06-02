import numpy as np
from src.decay_scheduler import decay_schedule
from tqdm import tqdm


def q_lambda(policy, environment, discount_factor=1.0, initial_learning_rate=0.5,
             minimum_learning_rate=0.01, learning_rate_decay_ratio=0.3, lambda_factor=0.3,
             number_of_episodes=500, episilon=0.1):
    """
    Perform Q(lambda) algorithm.
    
    Parameters:
        policy (function): Policy function to follow
        environment (object): OpenAI Gym-like environment object
        discount_factor (float): Discount factor for future rewards, default 1.0
        initial_learning_rate (float): Initial learning rate, default 0.5
        minimum_learning_rate (float): Minimum learning rate, default 0.01
        learning_rate_decay_ratio (float): Ratio to decay learning rate, default 0.3
        lambda_factor (float): Eligibility trace decay factor, default 0.3
        number_of_episodes (int): Total number of episodes to simulate, default 500
        episilon (float): Epsilon value for epsilon-greedy policy, default 0.1
    
    Returns:
        Q (numpy.ndarray): Final action-value function
        Q_track (numpy.ndarray): Evolution of action-value function over episodes
    """
    # Initialize the action-value function and the eligibility traces
    number_of_states = environment.observation_space.n
    number_of_actions = environment.action_space.n
    action_value_function = np.zeros((number_of_states, number_of_actions))
    action_value_function_track = np.zeros((number_of_episodes, number_of_states, number_of_actions))
    eligibility_traces = np.zeros((number_of_states, number_of_actions))

    # Create a learning rate decay schedule
    learning_rates = decay_schedule(initial_learning_rate, minimum_learning_rate,
                                    learning_rate_decay_ratio, number_of_episodes)
    
    # Create a epsilon decay schedule
    episilon_decay = decay_schedule(episilon, 0.01, 0.9, number_of_episodes)

    # Loop over each episode
    for episode in tqdm(range(number_of_episodes), leave=False):
        # Reset the eligibility traces
        eligibility_traces.fill(0)

        # Start a new episode and get the initial state
        observation, terminated, truncated = environment.reset(), False,False

        # Extract the current state from the tuple
        current_state = observation[0]

        # Get an initial action from the policy
        current_action = policy(action_value_function, current_state, episilon_decay[episode])

        # Continue looping until the episode is done
        while not (terminated or truncated):
            # Take a step in the environment with the action
            next_state, reward, terminated, truncated, info = environment.step(current_action)

            # Get the next action from the policy
            next_action = policy(action_value_function, current_state, episilon_decay[episode])

            # Calculate the Temporal Difference target
            td_target = reward + discount_factor * action_value_function[next_state, next_action] * (not terminated)

            # Calculate the Temporal Difference error
            td_error = td_target - action_value_function[current_state, current_action]

            # Increase the eligibility of the current state-action pair
            eligibility_traces[current_state, current_action] += 1

            # Update the action-value function using the TD error and the eligibility traces
            action_value_function += learning_rates[episode] * td_error * eligibility_traces

            # Decay the eligibility traces
            eligibility_traces *= discount_factor * lambda_factor
            # Update the current state and action
            current_state, current_action = next_state, next_action
        # Track the evolution of the action-value function
        action_value_function_track[episode] = action_value_function

    return action_value_function, action_value_function_track
