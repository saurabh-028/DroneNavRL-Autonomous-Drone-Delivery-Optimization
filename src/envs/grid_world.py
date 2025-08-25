
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DroneDeliveryEnv(gym.Env):
    """
    2D grid world with a drone, static no-fly rectangle, and N randomly moving birds.
    State: [drone_x, drone_y, bird1_x, bird1_y, ..., birdN_x, birdN_y]
    Actions: 0=up, 1=right, 2=down, 3=left
    Rewards: +100 goal, -100 collision or no-fly violation, -1 step cost
    Episode ends on goal, collision/violation, or max_steps.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(self, grid_size=10, num_birds=3, max_steps=300, no_fly_rect=((3,3),(5,5)), render_mode=None, seed=None):
        super().__init__()
        self.grid_size = grid_size
        self.num_birds = num_birds
        self.max_steps = max_steps
        self.no_fly_min = np.array(no_fly_rect[0], dtype=np.int32)
        self.no_fly_max = np.array(no_fly_rect[1], dtype=np.int32)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        obs_dim = 2 + 2 * self.num_birds
        low = np.zeros(obs_dim, dtype=np.int32)
        high = np.full(obs_dim, self.grid_size - 1, dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def _random_empty_cell(self, forbidden=set()):
        while True:
            cell = self.rng.integers(0, self.grid_size, size=2, dtype=np.int32)
            if tuple(cell) in forbidden:
                continue
            if self._in_no_fly(cell):
                continue
            return cell

    def _in_no_fly(self, pos):
        return np.all(pos >= self.no_fly_min) and np.all(pos <= self.no_fly_max)

    def _move_entity(self, pos, action):
        # 0=up,1=right,2=down,3=left
        delta = np.array([[-1,0],[0,1],[1,0],[0,-1]], dtype=np.int32)
        new = pos + delta[action]
        new = np.clip(new, 0, self.grid_size-1)
        return new

    def _random_bird_step(self, pos):
        a = int(self.rng.integers(0,5))  # 0..4 (4 = stay)
        if a == 4:
            return pos
        return self._move_entity(pos, a)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.steps = 0
        forbidden = set()
        self.start = np.array([0,0], dtype=np.int32)
        self.goal  = np.array([self.grid_size-1, self.grid_size-1], dtype=np.int32)

        self.drone = self.start.copy()
        forbidden.add(tuple(self.drone))
        forbidden.add(tuple(self.goal))

        self.birds = []
        for _ in range(self.num_birds):
            self.birds.append(self._random_empty_cell(forbidden))
            forbidden.add(tuple(self.birds[-1]))
        self.birds = np.array(self.birds, dtype=np.int32)

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        return np.concatenate([self.drone] + [b for b in self.birds], dtype=np.int32)

    def step(self, action):
        self.steps += 1
        # Move drone
        self.drone = self._move_entity(self.drone, action)

        # Move birds
        for i in range(self.num_birds):
            self.birds[i] = self._random_bird_step(self.birds[i])

        reward = -1.0
        terminated = False
        truncated = False

        # collision with bird
        for b in self.birds:
            if np.array_equal(self.drone, b):
                reward = -100.0
                terminated = True
                break

        # no-fly violation
        if not terminated and self._in_no_fly(self.drone):
            reward = -100.0
            terminated = True

        # goal
        if not terminated and np.array_equal(self.drone, self.goal):
            reward = 100.0
            terminated = True

        if self.steps >= self.max_steps and not terminated:
            truncated = True

        obs = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        size = 400
        cell = size // self.grid_size
        img = np.ones((size, size, 3), dtype=np.uint8) * 255

        # no-fly zone
        nf0, nf1 = self.no_fly_min, self.no_fly_max
        for x in range(nf0[0], nf1[0]+1):
            for y in range(nf0[1], nf1[1]+1):
                img[x*cell:(x+1)*cell, y*cell:(y+1)*cell] = (240, 200, 200)

        # grid lines
        for i in range(self.grid_size+1):
            img[i*cell:i*cell+1, :] = 0
            img[:, i*cell:i*cell+1] = 0

        # goal
        gx, gy = self.goal
        img[gx*cell:(gx+1)*cell, gy*cell:(gy+1)*cell] = (200, 255, 200)

        # birds
        for bx, by in self.birds:
            img[bx*cell:(bx+1)*cell, by*cell:(by+1)*cell] = (0, 0, 255)

        # drone
        dx, dy = self.drone
        img[dx*cell:(dx+1)*cell, dy*cell:(dy+1)*cell] = (255, 0, 0)

        return img
