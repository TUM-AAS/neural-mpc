import numpy as np


class GroundMap:
    def __init__(self, horizon=((-1, 1), (-1, 1)), resolution=0.1):
        assert ((horizon[0][1] - horizon[0][0]) / resolution).is_integer()
        assert ((horizon[1][1] - horizon[1][0]) / resolution).is_integer()
        self._horizon = horizon
        self._resolution = resolution

    def at(self, center: np.array):
        map_center_to_map_origin = np.array([self._horizon[0][0], self._horizon[1][0]])
        map_origin = center + map_center_to_map_origin
        return self.draw(map_origin - 0.5 * self._resolution, self.empty_map), map_origin

    def draw(self, pos, map):
        raise NotImplementedError

    @property
    def empty_map(self):
        x_len = int((self._horizon[0][1] - self._horizon[0][0]) / self._resolution) + 1
        y_len = int((self._horizon[1][1] - self._horizon[1][0]) / self._resolution) + 1
        empty_map = np.zeros((x_len, y_len))
        return empty_map


class GroundMapWithBox(GroundMap):
    def __init__(self, box_min, box_max, height, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._box_min = box_min
        self._box_max = box_max
        self._height = height

    def draw(self, pos, map):
        x_from = max(int((self._box_min[0] - pos[0]) // self._resolution), 0)
        x_to = min(int((self._box_max[0] - pos[0]) // self._resolution), map.shape[0] - 1)
        y_from = max(int((self._box_min[1] - pos[1]) // self._resolution), 0)
        y_to = min(int((self._box_max[1] - pos[1]) // self._resolution), map.shape[1] - 1)
        map[x_from:x_to, y_from:y_to] = self._height
        return map

if __name__ == '__main__':
    horizon = ((-7, 7), (-7, 7))
    box_min_ = (-4.25, 9.37)
    box_max_ = (-2.76, 10.13)
    box_height_ = 1.0

    map = GroundMapWithBox(box_min_, box_max_, box_height_, horizon=horizon)

    map_at, o = map.at(np.array([3, 9]))

    print(map_at)

    m_x = int((1.01 + o[0]) / 0.1)

    print(m_x)