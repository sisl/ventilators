import datetime
import pandas as pd
import os
import wandb
import datreq 

# Needed for wandb
# TODO put your wandb API key here
os.environ['WANDB_API_KEY'] = '8f4a9d3b7c2e1a65d091f7b6e385c2a491d6e437' 
# TODO put your wandb entity name here
WANDB_ENTITY = "johnsmith"

# Engine and data related imports
# High level docs: https://pulse.kitware.com/physeng.html#autotoc_md705
from pulse.engine.PulseEngine import PulseEngine
from pulse.cdm.engine import SEDataRequest, SEDataRequestManager

# Simulation parameters
SIM_HZ = 0.1
# How big is a timestep?
TIME_PER_ACTION_HR = 0.5
TIME_PER_ACTION_MIN = 60 * TIME_PER_ACTION_HR
TIME_PER_ACTION_S = 60 * TIME_PER_ACTION_MIN

# This is including the time component (first element of the state),
# note that the time component is not used when training learned models
SPO2_INDEX  = 8 - 1
PAO2_INDEX  = 12 - 1
AWRR_INDEX  = 10 - 1
HR_INDEX    = 5 - 1
IE_INDEX    = 26 - 1
PPLAT_INDEX = 27 - 1
PH_INDEX    = 28 - 1
# For drawing from normals
ACTION_GAUSSIAN_VARIANCES = [0.2,2,0.2,3,1,0.02]
# Absolute bounds on actions
ACTION_BOUNDS = [
    (0, 1),  # fio2
    (1, 30),  # pinsp
    (0.1, 3),  # ti
    (1, 30),  # rr
    (1, 25),  # peep
    (0, 1),  # slope
]

# These concern learning, so the state is without time (i.e. with
# time the state would be 26 elements long). This is to prevent the
# model learning some bias based on the time of the state in the simulation
DIM_STATE = 27
DIM_ACTION = 6
DIM_LATENT = 8

# Where are the dynamics models saved? 
FP_AUTOENCODER_CKPT     = "/mnt/learning/models2/autoencoder-2024-05-16-18-50-24.ckpt"
FP_LATENT_DYNAMICS_CKPT = "/mnt/learning/models2/dynamics-2024-05-16-19-05-13.ckpt"

def init_engine(verbose=True, set_fp_results=False, provided_fp_results=None):
    # The data we want to get back from the engine
    data_mgr = SEDataRequestManager(datreq.standard_data_requests())
    # Configure where and how data will be written to
    ts = get_timestamp()
    if set_fp_results:
        fp_results = provided_fp_results
    else:
        fp_results = f"/mnt/results/sim_{ts}/"
    # Ensure this folder exists
    if not os.path.exists(fp_results):
        os.makedirs(fp_results)
    fp_states = f"{fp_results}states.csv"
    fp_actions = f"{fp_results}actions.csv"
    data_mgr.set_results_filename(fp_states)
    data_mgr.set_samples_per_second(SIM_HZ) # Hz
    if verbose:
        print(f"Logging initialized")


    # Create the pulse engine and logging to folder
    pulse = PulseEngine()
    pulse.set_log_filename(f"{fp_results}pulse.log")
    if verbose:
        print(f"Pulse initialized at {SIM_HZ} Hz")

    return pulse, data_mgr, fp_results, fp_states, fp_actions

def get_timestamp():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  
    return timestamp

def make_actions_df_piecewise(df_actions, last_time):
    # The actions look like this:
    # fio2,pinsp,ti,rr,peep,slope,hold,Time(s)
    # 0,0,0,0,0,0,True,0.0
    # 1.0,0,0,0,24,0,True,599.9999999997443

    # But we want to plot actions like this:
    # fio2,pinsp,ti,rr,peep,slope,hold,Time(s)
    # 0,0,0,0,0,0,True,0.0
    # 0,0,0,0,0,0,True,599.9899999997443
    # 1.0,19,1.0,12.0,24,0.2,False,599.9999999997443
    # 1.0,19,1.0,12.0,24,0.2,False,1199.9899999991987

    # i.e. straight lines between actions. Do this
    # by adding the previous action just before
    # any new action/the end beyond the first one

    # Initialize a new DataFrame to store the modified actions
    df_modified_actions = pd.DataFrame(columns=df_actions.columns)

    # Iterate through each row in df_actions
    prev_action = None
    for i in range(len(df_actions)):
        action = df_actions.iloc[i]
        # Add the action to the midified action 
        # without using append
        # If the frame is empty just make it the first action
        if df_modified_actions.empty:
            df_modified_actions = pd.DataFrame(action).transpose()
        else:
            df_modified_actions = pd.concat([df_modified_actions, pd.DataFrame(action).transpose()])
        if prev_action is not None:
            prev_action['Time(s)'] = action['Time(s)'] - 0.0001
            df_modified_actions = pd.concat([df_modified_actions, pd.DataFrame(prev_action).transpose()])
        prev_action = action.copy()
    # Add the last action in at the final time
    prev_action['Time(s)'] = last_time
    df_modified_actions = pd.concat([df_modified_actions, pd.DataFrame(prev_action).transpose()])
    
    # Sort by time and reset the index
    df_modified_actions = df_modified_actions.sort_values(by='Time(s)') 
    df_modified_actions = df_modified_actions.reset_index(drop=True)

    # print(df_actions)
    # print(df_modified_actions)

    return df_modified_actions

def make_action_valid(action):
    """
    # fio2=0.21 - can't be outside 0-1
    # pinsp=13
    # ti=1.0,
    # rr=12.0,
    # peep=5.0 - can't be greater than 18
    # slope=0.1,
    
    FATAL:Positive End Expired Pressure cannot be higher than the Peak Inspiratory Pressure.
    FATAL:Inspiratory Period is longer than the total period applied using Respiration Rate.
    FATAL:Inspiration Waveform Period (i.e., Slope) cannot be longer than the Inspiratory Period.
    """
    # Ensure FiO2 is within the range [0, 1]
    action[0] = max(0.0, min(action[0], 1.0))  

    # Ensure PEEP is within the range [0, 18]
    action[4] = max(0.0, min(action[4], 18.0))  

    # Ensure that PEEP is not higher than the Peak Inspiratory Pressure (Pinsp)
    action[4] = min(action[4], action[1]*0.95)  # Adjust PEEP to match Pinsp if it's higher

    # Ensure that Inspiratory Period (Ti) is not longer than the total period applied using Respiration Rate (RR)
    action[2] = min(action[2], (60/action[3])*0.95)  # Adjust Ti to match RR if it's longer

    # Ensure that slope is not longer than the Inspiratory Period (Ti)
    action[5] = min(action[5], action[2]*0.95)  # Adjust slope to match Ti if it's longer

    return action

def write_reward_action_file(filepath, results):
    # Write out the results for this step to the step specific directory
    # First sort by reward (maximum first)
    results.sort(key=lambda x: x[1], reverse=True)
    with open(filepath, 'w') as f:
        f.write("reward, action\n")
        for action, reward in results:
            f.write(f"{reward}, {action}\n")

def init_new_wandb_run(project_name, name_prefix):
    run_name = f"{name_prefix}-{get_timestamp()}"
    # Finish old run, start new one
    wandb.finish()
    # and setup pytorch lightning logger (we use standard wandb
    # to log test images, and pytorch lightning's wrapper for 
    # use in other pytorch lightning callback functions).
    # It is important that they are logging to the same place
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=project_name,
        name=run_name,
        #mode="offline",
    )
    return run, run_name