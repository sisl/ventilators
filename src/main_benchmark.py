import sys
import os
import concurrent.futures

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

def benchmark_agent(agent_class, patient_config, wandb_project_name):
    
    # Create a ventilator environment 
    num_days = 2 # 48 hours
    env = environment.VentilatorEnvironment(
        patient_config=patient_config,
        timesteps_off_vent=0,
        timesteps_on_vent=int((24 / utils.TIME_PER_ACTION_HR) * num_days),
        #time_on_vent=1, # for benchmarking decision time
        wandb_project_name=wandb_project_name,
    )

    if agent_class == agents.AgentEmbed2Control or agent_class == agents.AgentEmbed2MPPI:
        # For MPC we need to provide the dynamics predictor
        num_samples = 1024
        horizon = 4
    else:
        num_samples = 32
        horizon = 4

    # Run the environment with the policy
    # and get the results
    agent = agent_class(
        env=env,
        # Not all agents need this
        num_samples=num_samples,
        horizon=horizon,
    )
    print(f"Simulating agent {agent_class}")
    env.simulate(agent)

def benchmark_all_agents(wandb_project_name):
    num_patients = 100
    num_half_patients = num_patients // 2 

    agents_to_benchmark = [
        agents.AgentEmbed2Control,
        agents.AgentEmbed2MPPI,
        agents.AgentRandom,
        agents.AgentARDSNet,
        agents.AgentHighFlow,
        agents.AgentSampleMPC,
    ]

    # Parallelize this
    #num_workers = min(16, os.cpu_count() + 4)
    num_workers = 1 # No parallelization
    print(f"Have {num_workers} worker(s) available for parallelization")

    # Distributes patients across sex and age
    def create_patient_config(i):
        age_high = 65
        age_low = 18
        age_diff = age_high - age_low
        age_step = age_diff / num_half_patients
        ages = [round(age_low + j * age_step) for j in range(num_half_patients)]
        return {
            "name": f"patient-{i}",
            "sex": "Male" if i < num_half_patients else "Female",
            "age": ages[i % num_half_patients],
            "weight": 150,  # lbs
            "height": 70,   # inches
        }
    
    # Perform benchmarking of a type of agent on a single patient
    def benchmark_single_agent(agent, i):
        patient_config = create_patient_config(i)
        benchmark_agent(
            agent, 
            patient_config,
            wandb_project_name=wandb_project_name,
        )

    # Now we can run the benchmarking across all patients for a given agent
    for agent in agents_to_benchmark:
        for i in range(num_patients):
            benchmark_single_agent(agent, i)

    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     futures = []
    #     print("Creating futures for benchmarking")
    #     for agent in agents_to_benchmark:
    #         for i in range(num_patients):
    #             futures.append(executor.submit(benchmark_single_agent, agent, i))

    #     # Optionally, wait for all futures to complete
    #     print("Waiting for futures to complete")
    #     concurrent.futures.wait(futures)

# python3 /mnt/src/main_benchmark.py
if __name__ == "__main__":
    # Provide the wandb project that we want to log to
    benchmark_all_agents(
        wandb_project_name="ventilators-evaluation"
    )
