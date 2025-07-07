import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import healthy_ranges
import utils

def reward_in_range_with_linear_penalty(
    patient_object,
    unit,
    number, 
    R_in_range=1,
    R_linear_penalty=1,#float('inf'),
):
    """
    linear penalty should be positive to penalise being outside the range
    """

    # Get the lower and upper bounds for the number for the patient
    lower_bound, upper_bound = healthy_ranges.get_healthy_range(
        unit=unit,
        patient_object=patient_object,
    )

    if lower_bound <= number <= upper_bound:
        # If the number is within the specified range, return a positive reward
        return R_in_range
    else:
        #return 0
        # If the number is outside the range, calculate a linearly decreasing reward
        if number < lower_bound:
            return R_in_range - R_linear_penalty * (lower_bound - number)
        else:
            return R_in_range - R_linear_penalty * (number - upper_bound)

def compute_action_reward(action):
    """
    Generally speaking we want to follow a philosophy of minimal intervention, so
    we want to minimze actions where possible
    """

    # See where we are in the valid region for each action type
    # and convert to 0-1 scale based on human interpretation
    def get01(action_index, action_val):
        # Get the bounds for the action
        lower_bound = utils.ACTION_BOUNDS[action_index][0]
        upper_bound = utils.ACTION_BOUNDS[action_index][1]
        # May be over or under
        if action_val < lower_bound:
            return 0
        if action_val > upper_bound:
            return 1
        # Convert to 0-1 scale
        return (action_val - lower_bound) / (upper_bound - lower_bound)
    action01s = [get01(i, action[i]) for i in range(len(action))]   
    
    # Sum and divide by action dimension so that it's in the range [0, 1]
    action_reward = sum(action01s) / len(action01s)

    # Then weight with a hyperparameter and return negated (more intervention is bad)
    hyperparameter = 0

    return -hyperparameter * action_reward

def compute_state_reward(state, patient_object):
    """
    The reward for a patient is associated with them being in healthy 
    physiological ranges (given their age/sex/etc.) across a number
    of metrics
    
    Note that the state in this case MAY or MAY NOT have a time component
    """

    # Check to see if we have a time component
    has_no_time = utils.DIM_STATE == len(state)
    if has_no_time:
        # If we have no time, we need to add it
        state = [0] + list(state)
    assert len(state) == (utils.DIM_STATE + 1), f"State has length {len(state)} but should have length {utils.DIM_STATE + 1}"

    # Get the useful info
    spo2 = state[utils.SPO2_INDEX]
    pao2 = state[utils.PAO2_INDEX]
    awrr = state[utils.AWRR_INDEX]
    hr = state[utils.HR_INDEX]
    ie = state[utils.IE_INDEX]
    pplat = state[utils.PPLAT_INDEX]
    ph = state[utils.PH_INDEX]
    
    # Oxygenation goal: PaO2 55-80 mmHg or SpO2 88-95%
    # SpO2 we measure directly
    # SpO2 given as a decimal, but PaO2 in mmHg
    R_spo2 = reward_in_range_with_linear_penalty(
        patient_object,
        "spo2",
        spo2,
    )
    R_pao2 = reward_in_range_with_linear_penalty(
        patient_object,
        "pao2",
        pao2,
    )
    # airway respiration rate
    R_awRR = reward_in_range_with_linear_penalty(
        patient_object,
        "awrr",
        awrr,
    )

    # Heart rate goal 
    R_hr = reward_in_range_with_linear_penalty(
        patient_object,
        "hr",
        hr,
    ) 
    
    # I/E ratio goal: (inspiration duration < expiration duration, so < 0.5 is desirable)
    R_ie = reward_in_range_with_linear_penalty(
        patient_object,
        "ie",
        ie,
    )
    
    # Plateau pressure goal: <= 30 cmH20
    R_pplat = reward_in_range_with_linear_penalty(
        patient_object,
        "pplat",
        pplat,
    )
    
    # pH goal: 7.3-7.45
    R_ph = reward_in_range_with_linear_penalty(
        patient_object,
        "ph",
        ph,
    )
    
    # Combine into final reward
    Rs = [R_spo2, R_pao2, R_awRR, R_hr, R_ie, R_pplat, R_ph]
    # TODO if we're going to hyperparameter weight then we should do it here
    #weights = [0.5, 3, 0.5, 3, 1] #, 1, 1]
    weights = [1,1,1,1,1,1,1]
    normalized_weights = [w / sum(weights) for w in weights]
    R = sum([Rs[i] * normalized_weights[i] for i in range(len(Rs))])

    return R, Rs