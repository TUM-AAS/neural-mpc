""" Executable script to train a custom Gaussian Process on recorded flight data.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""


import os
import time
import argparse
import subprocess
import numpy as np

import torch
from torch.utils.data import DataLoader
import ml_casadi.torch as mc

from tqdm import tqdm

from src.model_fitting.mlp_common import GPToMLPDataset, NormalizedMLP
from src.utils.utils import safe_mkdir_recursive, load_pickled_models
from src.utils.utils import get_model_dir_and_file
from src.model_fitting.gp_common import GPDataset, read_dataset
from config.configuration_parameters import ModelFitConfig as Conf



def main(x_features, u_features, reg_y_dims, model_ground_effect, quad_sim_options, dataset_name,
         x_cap, hist_bins, hist_thresh,
         model_name="model", epochs=100, batch_size=64, hidden_layers=1, hidden_size=32, plot=False):

    """
    Reads the dataset specified and trains a GP model or ensemble on it. The regressed variables is the time-derivative
    of one of the states.
    The feature and regressed variables are identified through the index number in the state vector. It is defined as:
    [0: p_x, 1: p_y, 2:, p_z, 3: q_w, 4: q_x, 5: q_y, 6: q_z, 7: v_x, 8: v_y, 9: v_z, 10: w_x, 11: w_y, 12: w_z]
    The input vector is also indexed:
    [0: u_0, 1: u_1, 2: u_2, 3: u_3].

    :param x_features: List of n regression feature indices from the quadrotor state indexing.
    :type x_features: list
    :param u_features: List of n' regression feature indices from the input state.
    :type u_features: list
    :param reg_y_dims: Indices of output dimension being regressed as the time-derivative.
    :type reg_y_dims: List
    :param dataset_name: Name of the dataset
    :param quad_sim_options: Dictionary of metadata of the dataset to be read.
    :param x_cap: cap value (in absolute value) for dataset pruning. Any input feature that exceeds this number in any
    dimension will be removed.
    :param hist_bins: Number of bins used for histogram pruning
    :param hist_thresh: Any bin with less data percentage than this number will be removed.
    :param model_name: Given name to the currently trained GP.
    """

    # Get git commit hash for saving the model
    git_version = ''
    try:
        git_version = subprocess.check_output(['git', 'describe', '--always']).strip().decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(e.returncode, e.output)
    print("The model will be saved using hash: %s" % git_version)

    gp_name_dict = {"git": git_version, "model_name": model_name, "params": quad_sim_options}
    save_file_path, save_file_name = get_model_dir_and_file(gp_name_dict)
    safe_mkdir_recursive(save_file_path)

    # #### DATASET LOADING #### #
    if isinstance(dataset_name, str):
        df_train = read_dataset(dataset_name, True, quad_sim_options)
        gp_dataset_train = GPDataset(df_train, x_features, u_features, reg_y_dims,
                                     cap=x_cap, n_bins=hist_bins, thresh=hist_thresh)
        gp_dataset_val = None
        try:
            df_val = read_dataset(dataset_name, False, quad_sim_options)
            gp_dataset_val = GPDataset(df_val, x_features, u_features, reg_y_dims,
                                       cap=x_cap, n_bins=hist_bins, thresh=hist_thresh)
        except:
            print('Could not find test dataset.')
    else:
        raise TypeError("dataset_name must be a string.")

    dataset_train = GPToMLPDataset(gp_dataset_train, ground_effect=model_ground_effect)
    x_mean, x_std, y_mean, y_std = dataset_train.stats()
    data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    input_dims = len(x_features) + len(u_features) + (9 + 4 if model_ground_effect else 0)
    mlp_model = mc.nn.MultiLayerPerceptron(input_dims, hidden_size, len(reg_y_dims), hidden_layers, 'Tanh')
    model = NormalizedMLP(mlp_model, torch.tensor(x_mean).float(), torch.tensor(x_std).float(),
                          torch.tensor(y_mean).float(), torch.tensor(y_std).float())
    print(f"Train Dataset has {len(dataset_train)} points.")

    if gp_dataset_val:
        dataset_val = GPToMLPDataset(gp_dataset_val, ground_effect=model_ground_effect)
        data_loader_val = DataLoader(dataset_val, batch_size=10*batch_size, shuffle=False, num_workers=0)
        print(f"Val Dataset has {len(dataset_val)} points.")

    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.")

    if torch.cuda.is_available():
        model = model.to('cuda:0')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_infos = []
    p_bar = tqdm(range(epochs))
    for i in p_bar:
        model.train()
        losses = []
        for x, y in data_loader:
            if torch.cuda.is_available():
                x = x.to('cuda:0')
                y = y.to('cuda:0')
            x = x.float()
            y = y.float()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = torch.square(y - y_pred).mean()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        losses_val = []
        if gp_dataset_val:
            with torch.no_grad():
                for x, y in data_loader_val:
                    if torch.cuda.is_available():
                        x = x.to('cuda:0')
                        y = y.to('cuda:0')
                    x = x.float()
                    y = y.float()
                    y_pred = model(x)
                    loss = torch.square(y - y_pred)

                    losses_val.append(loss.numpy())
        train_loss_info = np.mean(losses)
        loss_info = np.mean(np.vstack(losses_val)) if losses_val else np.mean(losses)
        loss_infos.append(loss_info)
        p_bar.set_description(f'Train Loss: {train_loss_info}, Val Loss {loss_info:.6f}')
        p_bar.refresh()

        save_dict = {
            'state_dict': model.state_dict(),
            'input_size': input_dims,
            'hidden_size': hidden_size,
            'output_size': len(reg_y_dims),
            'hidden_layers': hidden_layers
        }
        torch.save(save_dict, os.path.join(save_file_path, f'{save_file_name}.pt'))

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(loss_infos)
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default="1000",
                        help="Number of epochs to train the model.")

    parser.add_argument("--batch_size", type=int, default="64",
                        help="Batch Size.")

    parser.add_argument("--hidden_layers", type=int, default="1",
                        help="Number of hidden layers.")

    parser.add_argument("--hidden_size", type=int, default="32",
                        help="Size of hidden layers.")

    parser.add_argument("--model_name", type=str, default="",
                        help="Name assigned to the trained model.")

    parser.add_argument('--x', nargs='+', type=int, default=[7],
                        help='Regression X variables. Must be a list of integers between 0 and 12. Velocities xyz '
                             'correspond to indices 7, 8, 9.')

    parser.add_argument('--u', action="store_true",
                        help='Use the control as input to the model.')

    parser.add_argument("--y", nargs='+', type=int, default=[7],
                        help="Regression Y variable. Must be an integer between 0 and 12. Velocities xyz correspond to"
                             "indices 7, 8, 9.")

    parser.add_argument('--ge', action="store_true",
                        help='Use the ground map as input to the model.')

    parser.add_argument("--plot", dest="plot", action="store_true",
                        help="Plot the loss after training.")
    parser.set_defaults(plot=False)

    input_arguments = parser.parse_args()

    # Use vx, vy, vz as input features
    x_feats = input_arguments.x
    if input_arguments.u:
        u_feats = [0, 1, 2, 3]
    else:
        u_feats = []

    model_ground_effect = input_arguments.ge

    # Regression dimension
    y_regressed_dims = input_arguments.y
    model_name = input_arguments.model_name

    epochs = input_arguments.epochs
    batch_size = input_arguments.batch_size
    hidden_layers = input_arguments.hidden_layers
    hidden_size = input_arguments.hidden_size
    plot = input_arguments.plot

    ds_name = Conf.ds_name
    simulation_options = Conf.ds_metadata

    histogram_pruning_bins = Conf.histogram_bins
    histogram_pruning_threshold = Conf.histogram_threshold
    x_value_cap = Conf.velocity_cap

    main(x_feats, u_feats, y_regressed_dims, model_ground_effect, simulation_options, ds_name,
         x_value_cap, histogram_pruning_bins, histogram_pruning_threshold,
         model_name=model_name, epochs=epochs, batch_size=batch_size, hidden_layers=hidden_layers,
         hidden_size=hidden_size, plot=plot)
