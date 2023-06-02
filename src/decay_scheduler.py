import numpy as np

def decay_schedule(initial_value, minimum_value, decay_ratio, max_steps, log_start=-2, log_base=10):
    """
    Generates a decay schedule for a value over a given number of steps.

    Parameters:
    initial_value (float): The initial value of the schedule.
    minimum_value (float): The minimum value to decay towards.
    decay_ratio (float): The ratio of steps for which the decay should occur.
    max_steps (int): The maximum number of steps.
    log_start (int): The exponent for the start of the logarithmic space.
    log_base (int): The base of the logarithmic space.

    Returns:
    np.array: The decayed values over the specified number of steps.
    """

    # Calculate the number of steps for decay and remaining steps
    decay_steps = int(max_steps * decay_ratio)
    remaining_steps = max_steps - decay_steps

    # Generate logarithmic decay values in reverse order
    decay_values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]

    # Normalize decay values to range [0, 1]
    normalized_values = (decay_values - decay_values.min()) / (decay_values.max() - decay_values.min())

    # Scale normalized values to the desired range
    scaled_values = (initial_value - minimum_value) * normalized_values + minimum_value

    # Pad the decayed values with the minimum value for the remaining steps
    padded_values = np.pad(scaled_values, (0, remaining_steps), 'edge')

    return padded_values
