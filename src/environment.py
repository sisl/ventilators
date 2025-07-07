import csv
import os
import sys
import datetime
import time
import copy
import json
from tqdm import tqdm
import numpy as np
import importlib
import wandb
import matplotlib.pyplot as plt

# Intubation 
from pulse.cdm.patient_actions import eIntubationType, SEIntubation

# Mechanical ventilator (incl control schemes) imports
from pulse.cdm.mechanical_ventilator import eSwitch, eDriverWaveform
from pulse.cdm.mechanical_ventilator_actions import SEMechanicalVentilatorConfiguration, \
                                                    SEMechanicalVentilatorContinuousPositiveAirwayPressure, \
                                                    SEMechanicalVentilatorPressureControl, \
                                                    SEMechanicalVentilatorVolumeControl, \
                                                    SEMechanicalVentilatorHold, \
                                                    SEMechanicalVentilatorLeak, \
                                                    eMechanicalVentilator_PressureControlMode, \
                                                    eMechanicalVentilator_VolumeControlMode

# Units
from pulse.cdm.scalars import FrequencyUnit, PressureUnit, PressureTimePerVolumeUnit, \
                              TimeUnit, VolumeUnit, VolumePerPressureUnit, VolumePerTimeUnit, \
                              LengthUnit, MassUnit

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
from learning.eval import DynamicsPredictor

def simulate(
    pulse, 
    data_mgr, 
    fp_results, 
    fp_states, 
    fp_actions,
    pat,
    timesteps_off_vent,
    timesteps_on_vent,
    agent,
    wandb_run,
):    
    
    print("Beginning simulation")
    print(f"\t- timestep is {utils.TIME_PER_ACTION_S} seconds, or {utils.TIME_PER_ACTION_MIN:.2f} minutes, or {utils.TIME_PER_ACTION_HR:.2f} hours")
    print("\t- time_off_vent (steps):", timesteps_off_vent)
    print("\t- time_on_vent (steps):", timesteps_on_vent)
    print(f"\t- vent off for {timesteps_off_vent*utils.TIME_PER_ACTION_HR:.2f} hours")
    print(f"\t- vent on for {timesteps_on_vent*utils.TIME_PER_ACTION_HR:.2f} hours")

    # Give them ARDS
    # As per Jana's suggestion, ARDS in the 0.9 regime
    # is at the extremis of what you would see in real life,
    # so we won't ever go that high
    # 0.5 is a reasonable value.
    ards_setting = 0.5
    patient.ards(
        pulse,
        ards_left=ards_setting,
        ards_right=ards_setting,
    )

    # We need to intubate tracheally, otherwise a mask will be
    # assumed to be used instead
    # https://pulse.kitware.com/group___intubation_data__e_type.html
    intubation = SEIntubation()
    intubation.set_type(eIntubationType.Tracheal)
    pulse.process_action(intubation)

    # If testing dynamics predictor this can be used to advance state
    #learned_dynamics = DynamicsPredictor()
    #step_with_pulse_not_learned_dynamics = True

    # Advance the patient's condition on no ventilation to start
    for _ in tqdm(range(timesteps_off_vent), desc="Advancing without ventilation"):
        pulse.advance_time_s(utils.TIME_PER_ACTION_S)
    print("Patient is now on ventilation")

    # Do everything through the ventilator object
    vent = Ventilator(pulse)

    # Measure and plot accumulated rewards
    R_accumulated = 0

    # Now we advance until the end of the simulation
    for i in tqdm(range(timesteps_on_vent), desc="Taking ventilation actions"):

        # Get relevant physiological data
        data = pulse.pull_data()

        # Benchmark how long it takes for the policy to make a decision
        start_time = time.time()
        
        # Get the action from the policy
        action = agent.action(
            pat,
            data,
        )

        # Log the time taken to make a decision
        decision_time = time.time() - start_time
    
        # Update the ventilator (action)
        vent.update(*action)

        # Advance on this ventilation setting
        pulse.advance_time_s(utils.TIME_PER_ACTION_S)

        # Perform tracking
        R_state, R_state_components = rewards.compute_state_reward(data, pat)
        R_spo2, R_pao2, R_awrr, R_hr, R_ie, R_pplat, R_ph = R_state_components
        R_action = rewards.compute_action_reward(action)
        R_per_step = R_state + R_action
        # Add 100 to make it positive, note that this doesn't change the 
        # relative values of the rewards, so it makes no difference to the
        # policy/final result, apart from making the scores a little more
        # readable and making >0 scores more interpretable (i.e. they correspond
        # to good behavior)
        R_accumulated += (R_per_step + 100) 
        wandb_run.log({
            # Action
            "action/Fraction inspired O2": action[0],
            "action/Peak inspiratory pressure": action[1],
            "action/Inspiratory time": action[2],
            "action/Respiratory rate": action[3],
            "action/Positive end-expiratory pressure": action[4],
            "action/Slope": action[5],
            "action/Decision time (s)": decision_time,

            # Rewards
            "reward/Accumulated": R_accumulated,
            "reward/Per Step": R_per_step,
            "reward/State": R_state,
            "reward/Action": R_action,
            # State reward components. Order is sp02, pao2, awrr, hr, ie
            "reward/Ox. Saturation (Pulse Oximetry) SpO2": R_spo2,
            "reward/Pulmonary Arterial O2": R_pao2,
            "reward/Airway Respiratory Rate": R_awrr,
            "reward/Heart Rate": R_hr,
            "reward/I:E Ratio": R_ie,
            "reward/Plateau Pressure": R_pplat,
            "reward/Blood pH": R_ph,
        })
        
    # The simulation is complete!
        
    # States are automatically saved, but actions are not
    # Write the CSV file
    with open(fp_actions, 'w', newline='') as f:
        action_dicts = vent.get_actions()
        writer = csv.DictWriter(f, fieldnames=action_dicts[0].keys())
        # Write header
        writer.writeheader()
        # Write rows
        writer.writerows(action_dicts)

    print(f"Simulation complete! ({timesteps_off_vent + timesteps_on_vent} minutes total simulation time)")
    print(f"\t- states (pulse) saved to '{fp_states}'")
    print(f"\t- actions saved to '{fp_actions}'")
    
    return fp_results, fp_states, fp_actions

def run(
    pulse, 
    data_mgr, 
    fp_results, 
    fp_states, 
    fp_actions,
    pat,
    timesteps_off_vent,
    timesteps_on_vent,
    agent,
    wandb_project_name,
):
    
    # Initialize wandb logging
    wandb_run, _ = utils.init_new_wandb_run(
        project_name=wandb_project_name,
        name_prefix=agent.__class__.__name__,
    )

    # Log the patient, name sex age weight height
    wandb_run.log({
        #"patient/name": pat.name,
        "patient/sex": 0 if pat.sex == "Male" else 1, 
        "patient/age": pat.age,
        "patient/weight": pat.weight,
        "patient/height": pat.height,
    })

    # Run the simulation with this agent
    fp_results, fp_states, fp_actions = simulate(
        pulse, 
        data_mgr, 
        fp_results, 
        fp_states, 
        fp_actions,
        pat,
        timesteps_off_vent,
        timesteps_on_vent,
        agent,
        wandb_run,
    )

    # Plot all desired info and save         
    # Major plot                
    fig = plotting.plot_simulation(fp_states, fp_actions)
    fig.savefig(f"{fp_results}sim.jpg")

    # Snapshot (state checkins) 
    checks_per_sim = 6
    for i in range(checks_per_sim):
        t_start = i * (timesteps_off_vent+timesteps_on_vent)*utils.TIME_PER_ACTION_S/checks_per_sim
        t_end = t_start+15
        
        # Vitals information
        fig = plotting.plot_vitals(fp_states, t_start=t_start, t_end=t_end)
        fig.savefig(f"{fp_results}vitals_t={t_start}.jpg")

        # Ventilator information
        fig = plotting.plot_ventilator(fp_states, fp_actions, t_start=t_start, t_end=t_end)
        fig.savefig(f"{fp_results}vent_t={t_start}.jpg")

    # Close all figs
    plt.close('all')

class VentilatorEnvironment:
    def __init__(
        self,
        patient_config,
        timesteps_off_vent=5,
        timesteps_on_vent=25,
        wandb_project_name="ventilators-default",
    ):
        self.timesteps_off_vent = timesteps_off_vent
        self.timesteps_on_vent = timesteps_on_vent
        self.wandb_project_name = wandb_project_name

        # Create the logging utilities
        self.pulse, self.data_mgr, self.fp_results, self.fp_states, self.fp_actions = utils.init_engine()

        # Single patient per simulation
        self.patient_config = patient_config
        self.patient = Patient(
            self.pulse,
            self.data_mgr,
            name=patient_config["name"],
            sex=patient_config["sex"],
            age=patient_config["age"],
            weight=patient_config["weight"],
            height=patient_config["height"],
        )        

    def simulate(self, agent):
        """
        Simulate with an agent
        """
        
        # Report what agent we're using
        print(f"Simulating with {agent.__class__.__name__} agent")

        run(
            self.pulse, 
            self.data_mgr, 
            self.fp_results, 
            self.fp_states, 
            self.fp_actions,
            self.patient,
            self.timesteps_off_vent,
            self.timesteps_on_vent,
            agent,
            self.wandb_project_name,
        )