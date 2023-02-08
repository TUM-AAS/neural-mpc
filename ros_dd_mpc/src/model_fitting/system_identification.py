import argparse
import os

import numpy as np
import pandas as pd
import rosbag
import rospy

from config.configuration_parameters import ModelFitConfig as Conf
from src.experiments.point_tracking_and_record import make_record_dict
from src.quad_mpc.create_ros_dd_mpc import custom_quad_param_loader
from src.quad_mpc.quad_3d_mpc import Quad3DMPC
from src.utils.utils import jsonify, get_data_dir_and_file


def odometry_parse(odom_msg):
    p = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z]
    q = [odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
         odom_msg.pose.pose.orientation.z]
    v = [odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.z]
    w = [odom_msg.twist.twist.angular.x, odom_msg.twist.twist.angular.y, odom_msg.twist.twist.angular.z]

    return np.array(p + q + v + w)


def thrust_motor_model(motor_tau, thrust, thrust_des, dt):
    if motor_tau < 1e-12:
        return thrust_des
    tau_inv = 1 / motor_tau
    thrust_out = (
            tau_inv ** 2 * (thrust_des - 2 * (thrust * thrust_des) ** 0.5 + thrust) * dt ** 2
            + 2 * tau_inv * ((thrust * thrust_des) ** 0.5 - thrust) * dt
            + thrust
    )
    return thrust_out


def system_identification(quad_mpc, ds_name, sim_options):
    """
    Processes a rosbag with recorded odometry and send commands to do a system identification of the residual error.
    :param quad_mpc: Quad model used for forward simulation
    :param ds_name: Dataset name
    :param sim_options: Simulation options.
    :return:
    """
    # Open bag
    data_file = get_data_dir_and_file(ds_name, training_split=True, params=sim_options, read_only=True)
    if data_file is None:
        raise FileNotFoundError
    rec_file_dir, rec_file_name = data_file
    rec_file_name_bag = rec_file_name.replace('.csv', '.bag')
    rec_file = os.path.join(rec_file_dir, rec_file_name_bag)
    save_file = os.path.join(rec_file_dir, rec_file_name)
    bag = rosbag.Bag(rec_file)

    # Re-simulate thrust with motor model and communication delay
    thrust = None
    control = {'t': [], 'thrust': []}
    for topic, msg, t in bag.read_messages(topics=['control']):
        t = msg.header.stamp
        desired_thrust = np.array(msg.thrusts)
        if thrust is None:
            thrust = desired_thrust
        t = t + rospy.Duration.from_sec(quad_mpc.quad.comm_delay) + rospy.Duration.from_sec(0.001)
        for dt in np.arange(0.001, 0.021, step=0.001):
            thrust = thrust_motor_model(quad_mpc.quad.motor_tau, thrust, desired_thrust, 0.001)
            t_at = t + rospy.Duration.from_sec(dt)
            control['t'].append(t_at.to_sec())
            control['thrust'].append(thrust)

    control = pd.DataFrame(control)
    control = control.set_index('t')
    control = control[~control.index.duplicated(keep='last')]
    control = control.sort_index()

    state = {'t': [], 'state': []}
    states_list = []
    recording = False
    for topic, msg, t in bag.read_messages(topics=['recording_ctrl', 'state']):
        if topic == 'recording_ctrl':
            recording = msg.data
        if topic == 'state' and recording:
            t = msg.header.stamp
            state['t'].append(t.to_sec())
            x = odometry_parse(msg)
            state['state'].append(x)
            states_list.append(np.array(x))

    states_np = np.array(states_list)

    filtered_states_np = states_np.copy()
    from scipy.signal import savgol_filter
    window_size_xy = 31
    window_size_z = 31
    window_size_q = 31
    window_size_v = 31
    window_size_w = 121
    poly_order = 4
    filtered_states_np[:, :2] = savgol_filter(filtered_states_np[:, :2], window_size_xy, poly_order, axis=0)
    filtered_states_np[:, 2] = savgol_filter(filtered_states_np[:, 2], window_size_z, poly_order, axis=0)
    filtered_states_np[:, 3:7] = savgol_filter(filtered_states_np[:, 3:7], window_size_q, 2, axis=0)
    filtered_states_np[:, 3:7] = filtered_states_np[:, 3:7] / np.linalg.norm(filtered_states_np[:, 3:7], axis=1, keepdims=True)
    filtered_states_np[:, 7:10] = savgol_filter(filtered_states_np[:, 7:10], window_size_v, 2, axis=0)
    filtered_states_np[:, 10:] = savgol_filter(filtered_states_np[:, 10:], window_size_w, poly_order, axis=0)

    # def extract_windows(array, sub_window_size):
    #     examples = []
    #
    #     for i in range(0, array.shape[0]-sub_window_size, sub_window_size):
    #         example = array[i:sub_window_size + i]
    #         examples.append(np.expand_dims(example, 0))
    #
    #     return np.vstack(examples)
    #
    # from scipy.spatial.transform import Rotation as R
    # from scipy.spatial.transform import RotationSpline
    # window_rot = extract_windows(filtered_states_np[:, 3:7], 20)
    # mean_rot = []
    # for i in range(window_rot.shape[0]):
    #     mean_rot.append(R.from_quat(window_rot[i]).mean().as_quat())
    # mean_rot = np.vstack(mean_rot)
    # rotations = R.from_quat(mean_rot)
    # spline = RotationSpline(state['t'][9::20][:mean_rot.shape[0]], rotations)
    # filtered_states_np[:, 3:7] = spline(state['t']).as_quat()

    import matplotlib.pyplot as plt
    # plt.plot(states_np[:, 9])
    #plt.plot(filtered_states_np[3975:4025, 0])
    #plt.plot(states_np[3975:4025, 0])
    #plt.plot(np.gradient(filtered_states_np[3975:4025, 1])/1e-2)
    #plt.plot(filtered_states_np[3975:4025, 7])
    #plt.plot(filtered_states_np[1800:2000, 3])
    #plt.plot(states_np[1800:2000, 3])
    #plt.plot(filtered_states_np[3775:4025, 3:7] * np.sign(filtered_states_np[3775:4025, 3]))
    #plt.plot(filtered_states_np[3775:4025, 10])
    #plt.plot(states_np[3775:4025, 10])
    #plt.show()

    state = {'t': state['t'], 'state': []}
    for i in range(filtered_states_np.shape[0]):
        state['state'].append(filtered_states_np[i])

    rec_dict = make_record_dict(state_dim=13)
    for i in range(1, len(state['t'])):
        last_state_idx = control.index.get_loc(state['t'][i-1], method='nearest')
        curr_state_idx = control.index.get_loc(state['t'][i], method='nearest')
        u0 = control['thrust'].iloc[last_state_idx] / quad_mpc.quad.max_thrust
        u1 = control['thrust'].iloc[curr_state_idx] / quad_mpc.quad.max_thrust

        # Only use subsequent states with similar thrust state for system identification (within 1%)
        if np.all(np.abs(u0 - u1) < 0.01):
            x_0 = state['state'][i-1]
            x_f = state['state'][i]
            u = np.vstack([u0, u1]).mean(0)
            dt = state['t'][i] - state['t'][i-1]
            if dt > 0.015:
                continue
            x_pred, _ = quad_mpc.forward_prop(x_0, u, t_horizon=dt, use_gp=False)
            x_pred = x_pred[-1, np.newaxis, :]

            rec_dict['state_in'] = np.append(rec_dict['state_in'], x_0[np.newaxis, :], axis=0)
            rec_dict['input_in'] = np.append(rec_dict['input_in'], u[np.newaxis, :], axis=0)
            rec_dict['state_out'] = np.append(rec_dict['state_out'], x_f[np.newaxis, :], axis=0)
            rec_dict['state_ref'] = np.append(rec_dict['state_ref'], np.zeros_like(x_f[np.newaxis, :]), axis=0)
            rec_dict['timestamp'] = np.append(rec_dict['timestamp'], state['t'][i])
            rec_dict['dt'] = np.append(rec_dict['dt'], dt)
            rec_dict['state_pred'] = np.append(rec_dict['state_pred'], x_pred, axis=0)
            rec_dict['error'] = np.append(rec_dict['error'], x_f - x_pred, axis=0)

    # Save datasets
    for key in rec_dict.keys():
        print(key, " ", rec_dict[key].shape)
        rec_dict[key] = jsonify(rec_dict[key])
    df = pd.DataFrame(rec_dict)
    df.to_csv(save_file, index=True, header=True)
    return
    import matplotlib.pyplot as plt
    #plt.plot(states_np[:, 9])
    plt.plot(np.gradient(out[:, 8]))
    plt.show()
    return

    # Match each state to the corresponding thrust state of the quad
    rec_dict = make_record_dict(state_dim=13)
    recording = False
    last_state_msg = None
    for topic, msg, t in bag.read_messages(topics=['recording_ctrl', 'state']):
        if topic == 'recording_ctrl':
            recording = msg.data
            if not recording:
                last_state_msg = None
        if topic == 'state' and recording:
            if last_state_msg is not None:
                last_state_idx = control.index.get_loc(last_state_msg.header.stamp.to_sec(), method='nearest')
                curr_state_idx = control.index.get_loc(msg.header.stamp.to_sec(), method='nearest')
                u0 = control['thrust'].iloc[last_state_idx] / quad_mpc.quad.max_thrust
                u1 = control['thrust'].iloc[curr_state_idx] / quad_mpc.quad.max_thrust

                # Only use subsequent states with similar thrust state for system identification (within 1%)
                if np.all(np.abs(u0 - u1) < 0.01):
                    x_0 = odometry_parse(last_state_msg)
                    x_f = odometry_parse(msg)
                    u = np.vstack([u0, u1]).mean(0)
                    dt = msg.header.stamp.to_sec() - last_state_msg.header.stamp.to_sec()
                    x_pred, _ = quad_mpc.forward_prop(x_0, u, t_horizon=dt, use_gp=False)
                    x_pred = x_pred[-1, np.newaxis, :]

                    rec_dict['state_in'] = np.append(rec_dict['state_in'], x_0[np.newaxis, :], axis=0)
                    rec_dict['input_in'] = np.append(rec_dict['input_in'], u[np.newaxis, :], axis=0)
                    rec_dict['state_out'] = np.append(rec_dict['state_out'], x_f[np.newaxis, :], axis=0)
                    rec_dict['state_ref'] = np.append(rec_dict['state_ref'], np.zeros_like(x_f[np.newaxis, :]), axis=0)
                    rec_dict['timestamp'] = np.append(rec_dict['timestamp'], msg.header.stamp.to_sec())
                    rec_dict['dt'] = np.append(rec_dict['dt'], dt)
                    rec_dict['state_pred'] = np.append(rec_dict['state_pred'], x_pred, axis=0)
                    rec_dict['error'] = np.append(rec_dict['error'], x_f - x_pred, axis=0)

            last_state_msg = msg

    # Save datasets
    for key in rec_dict.keys():
        print(key, " ", rec_dict[key].shape)
        rec_dict[key] = jsonify(rec_dict[key])
    df = pd.DataFrame(rec_dict)
    df.to_csv(save_file, index=True, header=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--quad", type=str, default="kingfisher",
                        help="Name of the quad.")

    input_arguments = parser.parse_args()

    ds_name = Conf.ds_name
    simulation_options = Conf.ds_metadata

    quad = custom_quad_param_loader(input_arguments.quad)
    quad_mpc = Quad3DMPC(quad)

    system_identification(quad_mpc, ds_name, simulation_options)