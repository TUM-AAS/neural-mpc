# Real-Time Neural MPC

This repository contains the code for experiments associated to our paper 

```
Real-time Neural-MPC: Deep Learning Model Predictive Control for Quadrotors and Agile Robotic Platforms
```
[Arxiv Link](https://arxiv.org/pdf/2203.07747)

If you are looking for the ML-CasADi framework code you can find it [here](https://github.com/TUM-AAS/ml-casadi).

## Installation
### Checkout Submodules
```
git submodule update --init --recursive
```
### Acados
- Follow the [installation instructions](https://docs.acados.org/installation/index.html).
- Install the [Python Interface](https://docs.acados.org/python_interface/index.html).
- Ensure that `LD_LIBRARY_PATH` is set correctly (`DYLD_LIBRARY_PATH`on MacOS).
- Ensure that `ACADOS_SOURCE_DIR` is set correctly.

### Further Requirements
```
pip install -r requirements.txt
```

Make sure the ML-CasADi framework is part of the python path.
```
export PYTHONPATH="${PYTHONPATH}:<path-to-git>/ml-casadi"
```
Python 3.9 is recommended.

# Experiments
The provided code is based on the work of [Torrente et al.](https://github.com/uzh-rpg/data_driven_mpc) All functionality of the original code base is retained.

Change the working directory to
```
cd ros_dd_mpc
```
## Simulation
### Data Collection
Run the following script to collect a few minutes of flight samples
```
python src/experiments/point_tracking_and_record.py --recording --dataset_name simplified_sim_dataset --simulation_time 300
```

### Fitting a MLP Model
Edit the following variables of configuration file in `config/configuration_parameters.py` (class `ModelFitConfig`) so that the training script is referenced to the desired dataset. For redundancy, in order to identify the correct data file, we require to specify both the name of the dataset as well as the parameters used while acquiring the data.
In other words, you must input the simulator options used while running the previous python script. If you did not modify these variables earlier, you don't need to change anything this time as the default setting will work:
```
    # ## Dataset loading ## #
    ds_name = "simplified_sim_dataset"
    ds_metadata = {
        "noisy": True,
        "drag": True,
        "payload": False,
        "motor_noise": True
    }
```

The following command will train an MLP model with 4 hidden layers 64 neurons each to model the residual error on the velocities in x, y, and z direction (7, 8, 9 in the state).
We assign a name to the model for future referencing, e.g.: `simple_sim_mlp`
```
python src/model_fitting/mlp_fitting.py --model_name simple_sim_mlp --hidden_size 64 --hidden_layers 4 --x 7 8 9 --y 7 8 9 --epochs 100
```
The model will be saved under the directory `ros_dd_mpc/results/model_fitting/<git_hash>/`

### Fitting GP and RDRv
For instructions on how to fit a GP or RDRv model for comparison see the [here](https://github.com/uzh-rpg/data_driven_mpc)

### Test the Fitted Model
```
python src/experiments/trajectory_test.py --model_version <git_hash> --model_name simple_sim_mlp --model_typ mlp_approx
```
where the `model_type` argument can be one of `mlp_approx`(Real-time Neural MPC), `mlp` (Naive Integration), `gp` (Gaussian Process Model).

For a baseline comparison result run the same script without model parameters:
```
python src/experiments/trajectory_test.py
```

Multiple models can be compared at once via
```
python src/experiments/comparative_experiment.py --model_version <git_hash_1 git_hash_2 ...> --model_name <name_1 name_2 ...> --model_type <type_1 type_2> --fast
```

Results are saved in the `results/` folder.

### Citing

If you use this code in an academic context, please cite the following publication:

```
@article{salzmann2023neural,
  title={Real-time Neural-MPC: Deep Learning Model Predictive Control for Quadrotors and Agile Robotic Platforms},
  author={Salzmann, Tim and Kaufmann, Elia and Arrizabalaga, Jon and Pavone, Marco and Scaramuzza, Davide and Ryll, Markus},
  journal={arXiv preprint arXiv:2203.07747},
  year={2023}
}
```