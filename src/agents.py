import sys
import os
import numpy as np
import tempfile
import datetime
from tqdm import tqdm
import multiprocessing as mp

from pulse.cdm.io.patient import serialize_patient_from_file
from pulse.engine.PulseEngine import PulseEngine

# Custom imports
sys.path.append('/mnt/src/')
import utils
from vents import Ventilator
import rewards
import learning.eval
import healthy_ranges

# Create an agent class that has a policy function that
# takes the patient object and state and returns the action
class Agent:
    def __init__(self, env=None, pulse=None, fp_results=None, **kwargs):
        if env is not None:
            self.pulse = env.pulse
            self.fp_results = env.fp_results
        elif pulse is not None and fp_results is not None:
            self.pulse = pulse
            self.fp_results = fp_results
        else:
            raise ValueError("Either 'env' or BOTH 'pulse' and 'fp_results' must be provided.")

    def action(self, patient_object, state):
        return np.array(self.policy(patient_object, state))

    def policy(self):
        raise NotImplementedError("The 'policy' function must be implemented in subclasses.")

# ----------------------------------------------------------------
    
class AgentRandom(Agent):
    def policy(self, patient_object, state):
        return self.get_action_uniform()

    @staticmethod
    def get_action_uniform():
        action = [np.random.uniform(low, high) for low, high in utils.ACTION_BOUNDS]
        action = np.array(action)
        action = utils.make_action_valid(action)
        return action
    
    @staticmethod
    def get_action_gaussian():
        action_means = [np.mean([low, high]) for low, high in utils.ACTION_BOUNDS]
        action_stds = utils.ACTION_GAUSSIAN_VARIANCES^0.5
        action = [np.random.normal(loc=mean, scale=std) for mean, std in zip(action_means, action_stds)]
        action = np.array(action)
        action = utils.make_action_valid(action)
        return action

# ----------------------------------------------------------------

class AgentHighFlow(Agent):        
    def policy(self, patient_object, state):
        # fio2, pinsp, ti, rr, peep, slope
        #action = [0.9, 13, 1.0, 12.0, 18, 0.1] - for testing only
        action = healthy_ranges.get_high_flow_ventilator_settings(patient_object, state)
        action = utils.make_action_valid(action)
        return action
    
# ----------------------------------------------------------------

class AgentARDSNet(Agent):
    def policy(self, patient_object, state):
        # fio2, pinsp, ti, rr, peep, slope
        return self.get_action(patient_object, state)

    @staticmethod
    def get_noisy_action(patient_object, state):
        # Get the action
        action = AgentARDSNet.get_action(patient_object, state)

        # Gaussian sample around base action 
        variances = utils.ACTION_GAUSSIAN_VARIANCES
        # Different futures should have different notions of random
        noise = np.random.normal(scale=variances)
        action += noise 
        action = utils.make_action_valid(action)

        return action

    @staticmethod
    def get_action(patient_object, state):
        """
        Will give the patient parameters and compute
        the ARDSNet protocol for them to get the PEEP, FiO2, etc.
        
        Sex, age, weight, height
        Male/Female, [18, 65], lbs, inches
        """

        # List representation of state, note that we DO include the time index here
        state = list(state)

        # Need this from state
        spo2 = state[utils.SPO2_INDEX]
        pao2 = state[utils.PAO2_INDEX]

        # Get the required units
        sex, age, weight, height = patient_object.sex, patient_object.age, patient_object.weight, patient_object.height

        # Start by calculating predicted body weight (height
        # in inches but result in kg)
        if sex.lower() == "male":
            pbw = 50 + 2.3 * (height - 60)
        elif sex.lower() == "female":
            pbw = 45.5 + 2.3 * (height - 60)

        # Will have V_T at initially 8 ml/kg of PBW, but then
        # reduce ideally to 6 ml/kg of PBW
        vt = 8 * pbw

        # This is one strategy, we'll select from here
        # based on how far away we are from the target SpO2
        low_peep_high_fio2 = [
            [5, 0.3],
            [5, 0.4],
            [8, 0.4],
            [8, 0.5],
            [10, 0.5],
            [10, 0.6],
            [10, 0.7],
            [12, 0.7],
            [14, 0.7],
            [14, 0.8],
            [14, 0.9],
            [16, 0.9],
            [18, 0.9],
            #[21, 1.0],
        ]

        # Want PaO2 to be 55-80 mmHg or SpO2 88-95%, get
        # a distance from here in the range 0-1
        spo2_distance = abs(spo2*100 - 91.5) / 100
        pao2_distance = abs(pao2 - 67.5) / 67.5
        
        # Scale to index and select
        index = int(spo2_distance * len(low_peep_high_fio2))
        # Must be in range
        index = min(index, len(low_peep_high_fio2) - 1)
        index = max(index, 0)
        peep, fio2 = low_peep_high_fio2[index]

        #print(f"ARDSNet protocol: PaO2: {paO2}, SpO2: {spo2}, PEEP: {peep}, FiO2: {fio2}, VT: {vt}")

        # Return the vent settings
        # fio2, pinsp, ti, rr, peep, slope
        action = [ fio2, 13, 1.0, 12.0, peep, 0.1 ]
        action = np.array(action)
        action = utils.make_action_valid(action)
        return action

# ----------------------------------------------------------------
    
# Note that all the sampling classes need some concept of 'drawing' an
# action. This could be around ARDSnet, or using some random sampling strategy
# (e.g. those in chapter 13 in Mykel Kochenderfer's Algorithms for Optimization book)
def sample_n_actions(patient_object, state, sample_plan="random-uniform", n=1):
    """
    TODO: to use this function effectively we need to redesign the agents
    to be able to sample the action space all at once, and then use the 
    actions one at a time in different future simulations. A generator approach could be
    useful
    """
    
    if sample_plan == "ardsnet-gaussian":
        return [AgentARDSNet.get_noisy_action(patient_object, state) for _ in range(n)]
    elif sample_plan == "random-gaussian":
        return [AgentRandom.get_action_gaussian() for _ in range(n)]
    elif sample_plan == "random-uniform":
        return [AgentRandom.get_action_uniform() for _ in range(n)]
    # elif sample_plan == "uniform-projection-plan":
    #     pass
    
    # Throw an exception
    raise ValueError(f"Unknown sample plan '{sample_plan}'")

# ----------------------------------------------------------------

class AgentSampleMPC(Agent):
    def __init__(
            self, 
            env, 
            num_samples=8,
            horizon=64,
        ):
        # Initialise pulse and fp results
        super().__init__(env=env)

        # For mpc we'll be saving a bunch of patient states to the log folder,
        # so we'll make a subfolder for that
        self.log_folder_mpc = f"{self.fp_results}sample-mpc/"
        if not os.path.exists(self.log_folder_mpc):
            os.makedirs(self.log_folder_mpc)

        # At each step we're going to run forward N samples for H steps
        self.num_samples = num_samples
        self.horizon = horizon

        # For multiprocessing futures
        self.num_processes = 32

    def policy(self, patient_object, state):
        """
        Uses sampling-based model predictive control to
        select the best next action
        """

        verbose = False

        # Save the current state as a file which we can serialize from
        timestamp = utils.get_timestamp()
        # This saves the entire engine as a state
        log_folder_sim = f"{self.log_folder_mpc}step-{timestamp}/"
        # It is very important that this path does not have extra slashes in it
        initial_state_serialization_file = f"{log_folder_sim}serialized.json"
        self.pulse.serialize_to_file(initial_state_serialization_file)    

        # Simulate many futures using multiprocessing
        with mp.Pool(processes=self.num_processes) as pool:
            # predefine the arguments that will go to each process
            args = [(_, initial_state_serialization_file, patient_object, f"{log_folder_sim}future-{_}/", self.horizon, None) for _ in range(self.num_samples)]

            # Execute them simultaneously
            with tqdm(total=self.num_samples, desc=f"Simulating {self.num_samples} future(s) in sample MPC", disable=(not verbose)) as pbar:
                # And update on completion
                results = []
                for result in pool.starmap(
                    self.simulate_future,
                    args
                ):
                    results.append(result)
                    pbar.update(1)

        # Write out the results for this step to the step specific directory
        # First sort by reward (maximum first)
        utils.write_reward_action_file(f"{log_folder_sim}results.csv", results)

        # Find the best action based on rewards and return
        best_action, _ = max(results, key=lambda x: x[1])
        return utils.make_action_valid(best_action)

    @staticmethod
    def simulate_future(future_index, initial_state_serialization_file, patient_object, log_folder_sim, horizon, pbar):
        """
        Will simulate the future n_steps steps from the initial state
        serialization file using the given action
        """

        # Important! Each future should do something different, as should each call to 
        # the policy function, so include a timestamp
        seed = future_index + int(datetime.datetime.now().timestamp())
        np.random.seed(seed)
        
        # Create a NEW pulse engine within the simulation
        inner_pulse, data_mgr, fp_results, fp_states, fp_actions = \
            utils.init_engine(set_fp_results=True, provided_fp_results=log_folder_sim, verbose=False)

        # Initialise from file as in
        # https://gitlab.kitware.com/physiology/engine/-/blob/stable/src/python/pulse/howto/HowTo_EngineUse.py
        if not inner_pulse.serialize_from_file(initial_state_serialization_file, data_mgr):
            print(f"Unable to load initial state file at '{initial_state_serialization_file}'")
            return
        
        # As per usual we act through a ventilator
        inner_vent = Ventilator(inner_pulse)
        
        # We're now in a new future. We want to now sample an action *around* the ARDSNet policy
        # action, or just sample from action space in some intelligent way. We'll
        # then apply this action
        states = []
        actions = []
        for i in range(horizon):
            # Get the state at this time
            state = inner_pulse.pull_data()
            states.append(state)

            # What would ARDSnet do, plus some exploratory noise?
            action = sample_n_actions(patient_object, state, n=1)[0]

            # Note the action for later
            actions.append(action)

            # Apply it via the ventilator (in the inner simulation)
            inner_vent.update(*action)

            # Advance time
            # TODO this works but should be neatened up to remove
            # legacy variables
            change_vent_every_n_minutes = 1
            inner_pulse.advance_time_s(change_vent_every_n_minutes * utils.TIME_PER_ACTION_S)

            if pbar is not None:
                pbar.update(1)

        # Get the reward associated with this simulation's intermediate 
        # states and actions
        R_states = [rewards.compute_state_reward(state, patient_object)[0] for state in states]
        R_actions = [rewards.compute_action_reward(action) for action in actions]
        R = sum(R_states) + sum(R_actions)

        # We'll return the action and the reward so that we can
        # take the action associated with the greatest future reward
        return actions[0], R


# ----------------------------------------------------------------

class AgentEmbed2Control(Agent):
    def __init__(
            self, 
            env, 
            num_samples=8,
            horizon=64,
        ):
        # Initialise pulse and fp results
        super().__init__(env=env)

        # Make a place to store e2c related logs
        self.log_folder_e2c = f"{self.fp_results}e2c/"
        if not os.path.exists(self.log_folder_e2c):
            os.makedirs(self.log_folder_e2c)

        # At each step we're going to run forward N samples for H steps
        self.num_samples = num_samples
        # Note that the dynamics predictor is designe do predict 1 step
        # forward, not one sim update forward, so this is correct:
        self.horizon = horizon 

        # For multiprocessing futures
        self.num_processes = 32

        # We'll simulate samples forward using a learned model
        self.dynamics_predictor = learning.eval.DynamicsPredictor()

    def policy(self, patient_object, state):
        """
        Uses e2c to select the best next action
        """

        verbose = False

        # We aren't going to use pulse to simulate forward, we'll use a learned model
        # The learning was done without the time dimension, so we'll need to remove that
        initial_state = state[1:]

        # TODO this can be parallelised

        # For each sample, we'll select actions (about ardsnet) and simulate them forward to the horizon
        futures = []
        for i in tqdm(range(self.num_samples), desc="Simulating futures with E2C", disable=(not verbose)):
            last_state = initial_state
            # What are the actions taken and resultant states in this future?
            actions = []
            states = [last_state]

            for j in range(self.horizon):
                # What would ARDSnet do, plus some exploratory noise? 
                # Note that we need to add the initial time element back in
                action = sample_n_actions(patient_object, [0] + last_state, n=1)[0]

                # What state would this then give us? Recall that the dynamics
                # predictor was trained to predict the next step (not the next
                # simulation hertz update)
                next_state = self.dynamics_predictor.next_state(last_state, action)
                last_state = next_state

                # Track everything
                actions.append(action)
                states.append(next_state)

            # Save it in the futures
            futures.append((states, actions))

        # We now have a bunch of futures, and immediate actions.
        # Get the reward of the final state, and associate it with
        # the first action that sets off each future
        results = []
        for states, actions in futures:
            # Get the reward associated with this simulation's states
            R_states = [rewards.compute_state_reward(state, patient_object)[0] for state in states]
            # And the reward associated with each action taken
            R_actions = [rewards.compute_action_reward(action) for action in actions]
            R = sum(R_states) + sum(R_actions)
            results.append((actions[0], R))

        # Write out the results for this step to the step specific directory
        # First sort by reward (maximum first)
        utils.write_reward_action_file(f"{self.log_folder_e2c}results-{utils.get_timestamp()}.csv", results)
        
        # Find the best action based on rewards
        best_action, _ = max(results, key=lambda x: x[1])
        return utils.make_action_valid(best_action)

# ----------------------------------------------------------------

class AgentEmbed2MPPI(Agent):
    def __init__(
            self, 
            env, 
            num_samples=8,
            horizon=64,
            lambda_mppi=1.0,
        ):
        # Initialise pulse and fp results
        super().__init__(env=env)

        # Make a place to store e2mppic related logs
        self.log_folder = f"{self.fp_results}e2mppi/"
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

        self.num_samples = num_samples
        self.horizon = horizon 
        self.lambda_mppi = lambda_mppi

        # For multiprocessing futures
        self.num_processes = 32

        # We'll simulate samples forward using a learned model
        self.dynamics_predictor = learning.eval.DynamicsPredictor()

    # Overwrite the name of this class to include the lambda parameter
    def __str__(self):
        return f"AgentEmbed2MPPI-lambda={self.lambda_mppi}"

    def policy(self, patient_object, state):
        """
        Uses embedding to control alongside MPPI to select the best next action
        """

        verbose = False

        # We aren't going to use pulse to simulate forward, we'll use a learned model
        # The learning was done without the time dimension, so we'll need to remove that
        initial_state = state[1:]

        # TODO this can be highly parallelised

        # Now we're going to control with MPPI, so essentially we need to generate
        # random control sequences up to the horizon, and apply them and see what happens
        # It's up to us how we draw these (ARDSnet, grid sampling etc.)
        # Then we compute their cost and weight them all, and combine them according 
        # to the (lambda controlled) weight
        # We then execute the first action of the combined input sequence
        control_sequences = []
        for i in range(self.num_samples):
            control_sequence = []
            # Generate a random control sequence
            for j in range(self.horizon):
                # TODO remove inner loop and try different sampling strategies
                action = sample_n_actions(patient_object, [0] + initial_state, n=1)[0]
                control_sequence.append(action)
            control_sequences.append(control_sequence)
        
        # How would they work out?
        control_sequence_rewards = []
        for control_sequence in control_sequences:
            last_state = initial_state
            total_reward = 0
            for action in control_sequence:
                next_state = self.dynamics_predictor.next_state(last_state, action)
                last_state = next_state

                # State rewards
                R_state, _ = rewards.compute_state_reward(next_state, patient_object)
                total_reward += R_state

                # Action rewards
                R_action = rewards.compute_action_reward(action)
                total_reward += R_action

            control_sequence_rewards.append(total_reward)

        # Now we compute the weights. Formula is w_k = exp(-L R_k) / sum_j(exp(-L R_j))
        # Note that if lambda is +inf then we'll take the best action (the weighting is 'sharp')
        # Note that if lambda is -inf then we'll take a random action
        # Note that we're doing a reward (not cost) paradigm here, so we have + not -'s
        denom   = sum([np.exp(+self.lambda_mppi * R) for R in control_sequence_rewards])
        # Check for divide by zero, if so make it very small
        if denom == 0:
            denom = 1e-6
        weights = [np.exp(+self.lambda_mppi * R) / denom for R in control_sequence_rewards]

        # Write out the results for each decision to this step's logging directory
        results = []
        for i in range(self.num_samples):
            results.append((control_sequences[i][0], f"R={control_sequence_rewards[i]:.2f}, w={weights[i]:f}"))
        # Sort by highest weight first so that we can see the most contributing action
        results = sorted(results, key=lambda x: weights[i])
        utils.write_reward_action_file(f"{self.log_folder}results-{utils.get_timestamp()}.csv", results)

        # Now we combine the control sequences
        optimal_control_sequence = np.zeros((self.horizon, utils.DIM_ACTION))
        for i, control_sequence in enumerate(control_sequences):
            for j, action in enumerate(control_sequence):
                optimal_control_sequence[j] += weights[i] * action
        
        # Do the next immediate step
        return utils.make_action_valid(optimal_control_sequence[0])
        
