import sys
import pandas as pd 
import numpy as np
import math
import os

# Custom imports
sys.path.append('/mnt/src/')
import datreq
import plotting
import utils
from vents import Ventilator
import rewards
import patient
from patient import Patient
import agents
import environment

# TODO change this list to include the result DIRECTORIES you want to process
# into training data
dirs_to_process = [ f"/mnt/results/XXX/{x}/" for x in os.listdir("/mnt/results/XXX/") ]

# python3 /mnt/src/main_to_data.py
if __name__ == "__main__":
    """
    In this code we want to take a fp_results folder
    from a benchmarking run (or similar), then look at the
    actions taken and the states observed, and turn
    it into a series of state/action training pairs, or
    histories, so that we can do deep learning on it
    """

    def process_dir(fp_results):
        # What results are we using?
        final_folder_name = os.path.basename(os.path.normpath(fp_results))
        fp_output = f"/mnt/learning/training-data/{final_folder_name}/"
        # Ensure the data output exists
        os.makedirs(fp_output, exist_ok=True)

        # Relevant info is in states.csv and actions.csv
        fp_states = f"{fp_results}states.csv"
        fp_actions = f"{fp_results}actions.csv"    
        df_states = pd.read_csv(fp_states)
        df_actions = pd.read_csv(fp_actions)

        # Do not want to interpolate times, we actually want to select the
        # states that correspond to time points where we have actions
        time_key = "Time(s)"
        # What keys are available
        print(f"States keys: {df_states.keys()}")
        print(f"Actions keys: {df_actions.keys()}")
        print(f"*{time_key} will be removed from both dataframes")
        # Compute the hz from the states by looking at the difference
        # between the first two entries
        delta_t = df_states[time_key][1] - df_states[time_key][0]
        print(f"Delta t is {delta_t}")
        # What is the first and last state time?
        # Small epsilon here avoids a rounding error
        t_min = delta_t * 1.5 # in seconds #df_states[time_key].min()
        t_max = df_actions[time_key].max()

        # Now we want to round the times associated with the actions to the
        # nearest delta_t
        df_actions[time_key] = df_actions[time_key].apply(lambda x: round(x / delta_t) * delta_t)

        # Only consider the actions in this range
        def cut_within_times(df, t_min, t_max):
            # We just want the rows of the dataframe where the time column
            # is between t_min and t_max, but we need to convert the t_min
            # and t_max to timedelta
            t_min = pd.to_timedelta(t_min, unit='s')
            t_max = pd.to_timedelta(t_max, unit='s')
            df = df[(df[time_key] >= t_min) & (df[time_key] <= t_max)]
            return df
        # Need to do this for both states and actions so that we can compare
        df_states[time_key] = pd.to_timedelta(df_states[time_key], unit='s')
        df_actions[time_key] = pd.to_timedelta(df_actions[time_key], unit='s')
        df_actions = cut_within_times(df_actions, t_min, t_max)

        # Then we can get the states at these action-times
        df_states = df_states.set_index(time_key).loc[df_actions[time_key]].reset_index()
    
        # Remove the time columsn from both
        df_states = df_states.drop(columns=[time_key])
        df_actions = df_actions.drop(columns=[time_key])
        # Print info about the shapes of each dataframe
        print(f"States shape: {df_states.shape}")
        print(f"Actions shape: {df_actions.shape}")

        # We can now save the states and corresponding actions
        # as csvs 
        df_states.to_csv(f"{fp_output}states.csv", index=False)
        df_actions.to_csv(f"{fp_output}actions.csv", index=False)

    for dir_to_process in dirs_to_process:
        process_dir(dir_to_process)