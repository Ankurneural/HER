import numpy as np

class Bit_Flip_Env:
    """
    """

    def __init__(self, n=4, max_steps=50) -> None:
        """
        """
        self.n_bits = n
        self.max_steps = max_steps
        self.n_actions = n

        self.observation = self.reset()
        self.obervation_space = {'observation': np.empty((self.n_bits)),
                                'desired_state': np.empty((self.n_bits)),
                                'goal_state': np.empty((self.n_bits))}
    def reset(self):
        """
        """
        self._bits = np.array([np.random.randint(2) for _ in range(self.n_bits) ])
        self._desired_goal = np.array([np.random.randint(2) for _ in range(self.n_bits)])
        self._achieved_goal = self._bits.copy()
        self._steps = 0
        return np.concatenate([self._bits, self._achieved_goal, self._desired_goal])

    def compute_rewards(self, desired_state, achieved_state, info):
        """
        """
        return 0.0 if (desired_state==achieved_state).all() else -1.0

    def step(self, action):
        """
        """
        assert action <= self.n_bits, "Invalid action"
        bit_value = 1 if self._bits[action]==0 else 0
        self._bits[action] = bit_value
        info = {}
        self._achieved_goal = self._bits.copy()
        reward = self.compute_rewards(self._desired_goal, self._achieved_goal, {})
        self._steps += 1
        if reward==0.0 or self._steps >= self.max_steps:
            done = True
        else:
            done = False

        info["is_success"] = 1 if reward==0 else 0
        obs = np.concatenate([self._bits, self._achieved_goal, self._desired_goal])

        return obs, reward, done, info
    
    def action_space_sampler(self):
        """
        """
        return np.random.randint(0, self.n_bits)

    def render(self):
        """
        """
        for bit in self._bits:
            print(bit, end=' ')
        print('\n')
    

