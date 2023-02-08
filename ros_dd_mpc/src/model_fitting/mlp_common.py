import numpy as np
from torch.utils.data import Dataset

import ml_casadi.torch as mc
from config.configuration_parameters import GroundEffectMapConfig
from src.utils.ground_map import GroundMapWithBox


class GPToMLPDataset(Dataset):
    def __init__(self, gp_dataset, ground_effect=False):
        super().__init__()
        self.x = gp_dataset.x
        self.y = gp_dataset.y
        if ground_effect:
            self.x_raw = gp_dataset.x_raw[tuple(gp_dataset.pruned_idx)]
            assert self.x.shape[0] == self.x_raw.shape[0]
            map_conf = GroundEffectMapConfig
            ground_map = GroundMapWithBox(np.array(map_conf.box_min),
                                          np.array(map_conf.box_max),
                                          map_conf.box_height,
                                          horizon=map_conf.horizon,
                                          resolution=map_conf.resolution)
            self._map_res = map_conf.resolution
            self._static_ground_map, self._org_to_map_org = ground_map.at(np.array(map_conf.origin))
        else:
            self._static_ground_map = None

    def stats(self):
        if self._static_ground_map is None:
            return self.x.mean(axis=0), self.x.std(axis=0), self.y.mean(axis=0), self.y.std(axis=0)
        else:
            x_mean = np.hstack([self.x.mean(axis=0), np.zeros(9+4)])
            x_std = np.hstack([self.x.std(axis=0), np.ones(9+4)])
            return x_mean, x_std, self.y.mean(axis=0), self.y.std(axis=0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        if self._static_ground_map is None:
            return self.x[item], self.y[item]
        else:
            x, y, z = self.x_raw[item][:3]
            orientation = self.x_raw[item][3:7]
            x_idxs = np.floor((x - self._org_to_map_org[0]) / self._map_res).astype(int) - 1
            y_idxs = np.floor((y - self._org_to_map_org[1]) / self._map_res).astype(int) - 1
            ground_patch = self._static_ground_map[x_idxs:x_idxs+3, y_idxs:y_idxs+3]

            relative_ground_patch = 4 * (np.clip(z - ground_patch, 0, 0.5) - 0.25)

            flatten_relative_ground_patch = relative_ground_patch.flatten(order='F')

            ground_effect_in = np.hstack([flatten_relative_ground_patch, orientation*0])

            return np.hstack([self.x[item], ground_effect_in]), self.y[item]


class NormalizedMLP(mc.TorchMLCasadiModule):
    def __init__(self, model, x_mean, x_std, y_mean, y_std):
        super().__init__()
        self.model = model
        self.input_size = self.model.input_size
        self.output_size = self.model.output_size
        self.register_buffer('x_mean', x_mean)
        self.register_buffer('x_std', x_std)
        self.register_buffer('y_mean', y_mean)
        self.register_buffer('y_std', y_std)

    def forward(self, x):
        return (self.model((x - self.x_mean) / self.x_std) * self.y_std) + self.y_mean

    def cs_forward(self, x):
        return (self.model((x - self.x_mean.cpu().numpy()) / self.x_std.cpu().numpy()) * self.y_std.cpu().numpy()) + self.y_mean.cpu().numpy()


class QuadResidualModel(mc.TorchMLCasadiModule):
    def __init__(self, hidden_size, hidden_layers):
        super().__init__()
        self._input_size = 10
        self._output_size = 6
        self.force_model = mc.nn.MultiLayerPerceptron(7, hidden_size, 3, hidden_layers, 'Tanh')
        self.torque_model = mc.nn.MultiLayerPerceptron(10, hidden_size, 3, hidden_layers, 'Tanh')

    def forward(self, x):
        v = x[:, :3]
        w = x[:, 3:6]
        u = x[:, 6:]
        force_in = mc.hcat([v, u])
        torque_in = mc.hcat([v, w, u])
        force_out = self.force_model(force_in)
        torque_out = self.torque_model(torque_in)
        out = mc.hcat([force_out, torque_out])
        return out

    def cs_forward(self, x):
        v = x[:3]
        w = x[3:6]
        u = x[6:]
        force_in = mc.vcat([v, u])
        torque_in = mc.vcat([v, w, u])
        force_out = self.force_model(force_in)
        torque_out = self.torque_model(torque_in)
        out = mc.vcat([force_out, torque_out])
        return out
