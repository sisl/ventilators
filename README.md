# Optimal Control of Mechanical Ventilators with Learned Respiratory Dynamics

This repository contains the code for the paper "Optimal Control of Mechanical Ventilators with Learned Respiratory Dynamics":

```
@inproceedings{ward2024optimal,
    title={Optimal Control of Mechanical Ventilators with Learned Respiratory Dynamics},
    author={Ward, Isaac R. and Asmar, Dylan M. and Arief, Mansur and Mike, Jana Krystofova and Kochenderfer, Mykel J.},
    booktitle={Proceedings of the IEEE 37th International Symposium on Computer-Based Medical Systems (CBMS)},
    year={2024},
}
```

The code is organized as follows:

```
VENTILATORS/
├── data/                           # Pulse engine data, this should not be modified
├── engine/                         # Pulse engine source code, with modifications to the Dockerfile
├── learning/
│   ├── example_training_data/      # An example of what training data should look like
│   └── trained_models/             # 2 pretrained models (state autoencoder and latent dynamics predictor)
├── resources/                      # ARDSnet pamphlets and procedures
├── results/                        # Results from inside the container will be saved to this mounted directory
├── src/
│   ├── examples/                   # Some useful examples for working with the Pulse engine
│   ├── learning/                   # Code pertaining to learning-based models, training, and evaluation
│   ├── patients/                   # Standard patient files
│   ├── agents.py                   # All agents used in the experiments
│   ├── datreq.py                   # Specifies what data will be returned from the Pulse engine
│   ├── environment.py              # A wrapper for the Pulse engine that allows for easy simulation
│   ├── healthy_ranges.py           # The healthy ranges for various patient variables
│   ├── main_benchmark.py           # Runs agents on the ventilator benchmark and saves the results
│   ├── main_to_data.py             # Converts data produced by benchmarking into training data
│   ├── main_train.py               # Trains the dynamics predictor and (optionally) the autoencoder
│   ├── patient.py                  # A wrapper for the Pulse engine that allows for easy interaction with a patient
│   ├── plotting.py                 # Code for plotting simulation results
│   ├── rewards.py                  # Specifies the reward function for the agents
│   ├── utils.py                    
│   └── vents.py                    # A wrapper for the Pulse engine that allows for easy interaction with a ventilator
├── .gitignore
├── docker-compose.yml
└── README.md
```

## Running this code

To run the code, you will need to have Docker installed on your machine. You can install Docker from [here](https://docs.docker.com/engine/install/). Once installed, use the file ```docker-compose.yml``` to build the Docker image and run the container. The following commands will build the image, run the container (and install all required python packages), and open a bash shell in the container:

```bash
docker-compose build --no-cache
docker-compose up -d --no-deps
docker exec -it ventilators bash
```

Note that the file ```docker-compose.yml``` mounts directories to the container at ```/mnt```. This allows you to edit the code on your local machine and run it in the container. **It also allows GPUs to be used in the container - if your machine does not have GPUs then edit the 'deploy' section of ```docker-compose.yml``` accordingly (comment it out)**. Inspect ```engine/Dockerfile``` to see how Pulse is built and how packages are installed. To run the code, you can use the following commands:

```bash
# Training autoencoder + dynamics predictor 
# (note that you won't need to do this if you're using the provided models)
python3 /mnt/src/main_train.py
# Benchmarking models/generating data
python3 /mnt/src/main_benchmark.py
# Formatting generated data into training data
python3 /mnt/src/main_to_data.py
```

Important notes:
- The full data used to train the provided models (see ```learning/trained_models/```) could not be included in this repository due to size constraints. However, the random rollout data can be generated using ```main_benchmark.py```.
- We use [Weights and Biases](https://wandb.ai/) to log all results. To run this code you'll need to sign up for weights and biases and replace lines 9-11 in ```src/utils.py``` with your own API key and entity name.

## Useful links

The Pulse engine has tutorials on how to use the engine in Python, here are the tutorials on mechanical ventilation:
- https://gitlab.kitware.com/physiology/engine/-/blob/stable/src/python/pulse/howto/HowTo_MechanicalVentilation.py
- https://gitlab.kitware.com/physiology/engine/-/blob/stable/src/python/pulse/howto/HowTo_MechanicalVentilator.py

If you want to build your own version of Pulse (build instructions here: https://gitlab.kitware.com/physiology/engine), then note that data must be copied over like so https://gitlab.kitware.com/physiology/jupyter/-/blob/master/Dockerfile?ref_type=heads
