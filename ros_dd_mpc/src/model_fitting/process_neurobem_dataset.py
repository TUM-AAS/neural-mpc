import argparse
import os
import random

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm

from config.configuration_parameters import DirectoryConfig
from config.configuration_parameters import ModelFitConfig as Conf
from src.experiments.point_tracking_and_record import make_record_dict, jsonify
from src.quad_mpc.create_ros_dd_mpc import custom_quad_param_loader
from src.quad_mpc.quad_3d_mpc import Quad3DMPC
from src.utils.utils import safe_mkdir_recursive, v_dot_q

val_files = ['merged_2021-02-03-16-58-13_seg_2.csv']

def main(quad):
    full_path = os.path.join(DirectoryConfig.DATA_DIR, 'neurobem_processed_data_cmd')
    files = os.listdir(full_path)
    random.shuffle(files)

    rec_dicts = []
    for file in tqdm(files):
        if file in val_files:
            print('Skipping Validation File')
            continue
        try:
            rec_dict = make_record_dict(state_dim=13)
            process_file(os.path.join(full_path, file), quad, rec_dict)
            rec_dicts.append(rec_dict)
        except Exception as e:
            print(e)

    rec_dict = {}
    rec_dict['state_in'] = np.concatenate([data_dict['state_in'] for data_dict in rec_dicts], axis=0)
    rec_dict['input_in'] = np.concatenate([data_dict['input_in'] for data_dict in rec_dicts], axis=0)
    rec_dict['state_out'] = np.concatenate([data_dict['state_out'] for data_dict in rec_dicts], axis=0)
    rec_dict['state_ref'] = np.concatenate([data_dict['state_ref'] for data_dict in rec_dicts], axis=0)
    rec_dict['timestamp'] = np.concatenate([data_dict['timestamp'] for data_dict in rec_dicts], axis=0)
    rec_dict['dt'] = np.concatenate([data_dict['dt'] for data_dict in rec_dicts], axis=0)
    rec_dict['state_pred'] = np.concatenate([data_dict['state_pred'] for data_dict in rec_dicts], axis=0)
    rec_dict['error'] = np.concatenate([data_dict['error'] for data_dict in rec_dicts], axis=0)

    del rec_dicts

    # Save datasets
    save_file_folder = os.path.join(DirectoryConfig.DATA_DIR, 'neurobem_dataset', 'train')
    safe_mkdir_recursive(save_file_folder)
    save_file = os.path.join(save_file_folder, f'npz_dataset_001.npz')
    np.savez(save_file, **rec_dict)


    # # Validation
    # rec_dicts = []
    # for file in tqdm(files[-20:]):
    #     try:
    #         rec_dict = make_record_dict(state_dim=13)
    #         process_file(os.path.join(full_path, file), quad, rec_dict)
    #         rec_dicts.append(rec_dict)
    #     except Exception as e:
    #         print(e)
    #
    # rec_dict = {}
    # rec_dict['state_in'] = np.concatenate([data_dict['state_in'] for data_dict in rec_dicts], axis=0)
    # rec_dict['input_in'] = np.concatenate([data_dict['input_in'] for data_dict in rec_dicts], axis=0)
    # rec_dict['state_out'] = np.concatenate([data_dict['state_out'] for data_dict in rec_dicts], axis=0)
    # rec_dict['state_ref'] = np.concatenate([data_dict['state_ref'] for data_dict in rec_dicts], axis=0)
    # rec_dict['timestamp'] = np.concatenate([data_dict['timestamp'] for data_dict in rec_dicts], axis=0)
    # rec_dict['dt'] = np.concatenate([data_dict['dt'] for data_dict in rec_dicts], axis=0)
    # rec_dict['state_pred'] = np.concatenate([data_dict['state_pred'] for data_dict in rec_dicts], axis=0)
    # rec_dict['error'] = np.concatenate([data_dict['error'] for data_dict in rec_dicts], axis=0)
    #
    # del rec_dicts
    #
    # # Save datasets
    # save_file_folder = os.path.join(DirectoryConfig.DATA_DIR, 'neurobem_dataset', 'test')
    # safe_mkdir_recursive(save_file_folder)
    # save_file = os.path.join(save_file_folder, f'npz_dataset_001.npz')
    # np.savez(save_file, **rec_dict)

def quaternion_to_euler(q):
    yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] - q[:, 1] * q[:, 2]),
        1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    pitch = np.arcsin(2 * (q[:, 0] * q[:, 2] + q[:, 3] * q[:, 1]))
    roll = np.arctan2(2 * (q[:, 0] * q[:, 1] - q[:, 2] * q[:, 3]),
        1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))

    return np.stack([yaw, pitch, roll], axis=-1)

def val(quad):
    full_path = os.path.join(DirectoryConfig.DATA_DIR, 'neurobem_processed_data_cmd')
    #val_files = ['merged_2021-02-23-22-54-17_seg_1.csv', 'merged_2021-02-23-17-35-26_seg_1.csv'] # 'merged_2021-02-18-18-09-47_seg_1.csv',

    rec_dicts = []
    for file in tqdm(val_files):
        try:
            rec_dict = make_record_dict(state_dim=13)
            process_file(os.path.join(full_path, file), quad, rec_dict)
            rec_dicts.append(rec_dict)
        except Exception as e:
            print(e)

    rec_dict = {}
    rec_dict['state_in'] = np.concatenate([data_dict['state_in'] for data_dict in rec_dicts], axis=0)
    rec_dict['input_in'] = np.concatenate([data_dict['input_in'] for data_dict in rec_dicts], axis=0)
    rec_dict['state_out'] = np.concatenate([data_dict['state_out'] for data_dict in rec_dicts], axis=0)
    rec_dict['state_ref'] = np.concatenate([data_dict['state_ref'] for data_dict in rec_dicts], axis=0)
    rec_dict['timestamp'] = np.concatenate([data_dict['timestamp'] for data_dict in rec_dicts], axis=0)
    rec_dict['dt'] = np.concatenate([data_dict['dt'] for data_dict in rec_dicts], axis=0)
    rec_dict['state_pred'] = np.concatenate([data_dict['state_pred'] for data_dict in rec_dicts], axis=0)
    rec_dict['error'] = np.concatenate([data_dict['error'] for data_dict in rec_dicts], axis=0)

    del rec_dicts

    # Save datasets
    save_file_folder = os.path.join(DirectoryConfig.DATA_DIR, 'neurobem_dataset', 'test')
    safe_mkdir_recursive(save_file_folder)
    save_file = os.path.join(save_file_folder, f'npz_dataset_001.npz')
    np.savez(save_file, **rec_dict)

def process_file(file_path, quad, rec_dict):
    data = pd.read_csv(file_path, encoding='latin-1')
    for x_0, x_f, u, dt in consecutive_data_points(data):
        resimulate(x_0, x_f, u, dt, quad, rec_dict)


def consecutive_data_points(data):
    for i in range(25, len(data)-1, 1):
        data_0 = data.iloc[i]
        data_1 = data.iloc[i+1]
        data_c = data.iloc[i]  # Communication Delay
        t0 = data_0['t']
        t1 = data_1['t']
        dt = t1 - t0

        x_0 = np.hstack([
            data_0['pos x'],
            data_0['pos y'],
            data_0['pos z'],
            data_0['quat w'],
            data_0['quat x'],
            data_0['quat y'],
            data_0['quat z'],
            data_0['vel x'],
            data_0['vel y'],
            data_0['vel z'],
            data_0['ang vel x'],
            data_0['ang vel y'],
            data_0['ang vel z'],
        ])

        x_1 = np.hstack([
            data_1['pos x'],
            data_1['pos y'],
            data_1['pos z'],
            data_1['quat w'],
            data_1['quat x'],
            data_1['quat y'],
            data_1['quat z'],
            data_1['vel x'],
            data_1['vel y'],
            data_1['vel z'],
            data_1['ang vel x'],
            data_1['ang vel y'],
            data_1['ang vel z'],
        ])

        # Velocity to world coordinate frame
        x_0[7:10] = v_dot_q(x_0[7:10], x_0[3:7])
        x_1[7:10] = v_dot_q(x_1[7:10], x_1[3:7])

        # Solve single rotor thrusts
        Jm1 = 1 / quad.J

        a1 = Jm1[0] * quad.y_f
        b1 = data_0['ang acc x'] - Jm1[0] * (quad.J[1] - quad.J[2]) * x_0[11] * x_0[12]
        a2 = -Jm1[1] * quad.x_f
        b2 = data_0['ang acc y'] - Jm1[1] * (quad.J[2] - quad.J[0]) * x_0[12] * x_0[10]
        a3 = Jm1[2] * quad.z_l_tau
        b3 = data_0['ang acc z'] - Jm1[2] * (quad.J[0] - quad.J[1]) * x_0[10] * x_0[11]
        a4 = np.ones(4,)
        b4 = data_c['cmd_thrust'] * quad_mpc.quad.mass

        u = np.linalg.solve([a1, a2, a3, a4], [b1, b2, b3, b4]) / quad.max_thrust

        # ypr_0 = quaternion_to_euler(x_0[np.newaxis, 3:7])[0][::-1]
        # x_0[12] = 1.0 * x_0[10] + np.sin(ypr_0[0]) * np.tan(ypr_0[1]) * x_0[11] + np.cos(ypr_0[0]) * np.tan(ypr_0[1]) * x_0[12]
        # x_0[11] = np.cos(ypr_0[0]) * x_0[11] - np.sin(ypr_0[0]) * x_0[12]
        # x_0[10] = np.sin(ypr_0[0]) / np.cos(ypr_0[1]) * x_0[11] + np.cos(ypr_0[0]) / np.cos(ypr_0[1]) * x_0[12]
        #
        # ypr_1 = quaternion_to_euler(x_1[np.newaxis, 3:7])[0][::-1]
        # x_1[10] = 1.0 * x_1[10] + np.sin(ypr_1[0]) * np.tan(ypr_1[1]) * x_1[11] + np.cos(ypr_1[0]) * np.tan(ypr_1[1]) * \
        #           x_1[12]
        # x_1[11] = np.cos(ypr_1[0]) * x_1[11] - np.sin(ypr_1[0]) * x_1[12]
        # x_1[12] = np.sin(ypr_1[0]) / np.cos(ypr_1[1]) * x_1[11] + np.cos(ypr_1[0]) / np.cos(ypr_1[1]) * x_1[12]

        # ypr_0 = quaternion_to_euler(x_0[np.newaxis, 3:7])[0][::-1]
        # x_0[10] = 1.0 * x_0[10] - np.sin(ypr_0[1]) * x_0[12]
        # x_0[11] = np.cos(ypr_0[0]) * x_0[11] + np.sin(ypr_0[0]) * np.cos(ypr_0[1]) * x_0[12]
        # x_0[12] = -np.sin(ypr_0[0]) * x_0[11] + np.cos(ypr_0[0]) * np.cos(ypr_0[1]) * x_0[12]
        #
        # ypr_1 = quaternion_to_euler(x_1[np.newaxis, 3:7])[0][::-1]
        # x_1[10] = 1.0 * x_1[10] - np.sin(ypr_1[1]) * x_1[12]
        # x_1[11] = np.cos(ypr_1[0]) * x_1[11] + np.sin(ypr_1[0]) * np.cos(ypr_1[1]) * x_1[12]
        # x_1[12] = -np.sin(ypr_1[0]) * x_1[11] + np.cos(ypr_1[0]) * np.cos(ypr_1[1]) * x_1[12]

        # u = np.hstack([
        #     data_0['mot 2'],
        #     data_0['mot 3'],
        #     data_0['mot 1'],
        #     data_0['mot 4'],
        # ])
        #
        # u = u ** 2 * quad.thrust_map[0] / quad.max_thrust

        yield x_0, x_1, u, dt

def f_rate(x, u, quad):
    """
    Time-derivative of the angular rate
    :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
    :param u: control input vector (4-dimensional): [trust_motor_1, ..., thrust_motor_4]
    :param t_d: disturbance torque (3D)
    :return: angular rate differential increment (scalar): dr/dt
    """

    rate = x[10:]
    return np.array([
        1 / quad.J[0] * (u.dot(quad.y_f) + (quad.J[1] - quad.J[2]) * rate[1] * rate[2]),
        1 / quad.J[1] * (-u.dot(quad.x_f) + (quad.J[2] - quad.J[0]) * rate[2] * rate[0]),
        1 / quad.J[2] * (u.dot(quad.z_l_tau) + (quad.J[0] - quad.J[1]) * rate[0] * rate[1])
    ]).squeeze()

def resimulate(x_0, x_f, u, dt, quad_mpc, rec_dict):
    x_pred, _ = quad_mpc.forward_prop(x_0, u, t_horizon=dt, use_gp=False)
    x_pred = x_pred[-1, np.newaxis, :]

    rec_dict['state_in'] = np.append(rec_dict['state_in'], x_0[np.newaxis, :], axis=0)
    rec_dict['input_in'] = np.append(rec_dict['input_in'], u[np.newaxis, :], axis=0)
    rec_dict['state_out'] = np.append(rec_dict['state_out'], x_f[np.newaxis, :], axis=0)
    rec_dict['state_ref'] = np.append(rec_dict['state_ref'], np.zeros_like(x_f[np.newaxis, :]), axis=0)
    rec_dict['timestamp'] = np.append(rec_dict['timestamp'], np.zeros_like(dt))
    rec_dict['dt'] = np.append(rec_dict['dt'], dt)
    rec_dict['state_pred'] = np.append(rec_dict['state_pred'], x_pred, axis=0)
    rec_dict['error'] = np.append(rec_dict['error'], x_f - x_pred, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--quad", type=str, default="kingfisher",
                        help="Name of the quad.")

    input_arguments = parser.parse_args()

    ds_name = Conf.ds_name
    simulation_options = Conf.ds_metadata

    quad = custom_quad_param_loader(input_arguments.quad)
    quad_mpc = Quad3DMPC(quad)

    main(quad_mpc)
    val(quad_mpc)
