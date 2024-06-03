import torch
import numpy as np
import gym
from botorch.test_functions import Ackley

from .base_benchmark import BaseBenchmark

from camtune.utils.logger import print_log
from camtune.utils.vars import DEVICE, DTYPE


class Lunarlanding(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.negate = False

        self.eval_counter = 0
        self.env = gym.make('LunarLander-v2')
        self.render = False

        self._obj_func = self

        self.__post_init__()
        
    def heuristic_control(self, s, w):
        angle_targ = s[0] * w[0] + s[2] * w[1]
        if angle_targ > w[2]:
            angle_targ = w[2]
        if angle_targ < -w[2]:
            angle_targ = -w[2]
        hover_targ = w[3] * np.abs(s[0])

        angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
        hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

        if s[6] or s[7]:
            angle_todo = w[8]
            hover_todo = -(s[3]) * w[9]

        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
            a = 2
        elif angle_todo < -w[11]:
            a = 3
        elif angle_todo > +w[11]:
            a = 1
        return a
        
    def __call__(self, x: torch.Tensor):
        self.eval_counter += 1
        x = x.detach().cpu().numpy()
    
        total_rewards = []
        for i in range(0, 3): # controls the number of episode/plays per trial
            state, _ = self.env.reset()
            rewards_for_episode = []
            num_steps = 2000
        
            for step in range(num_steps):
                if self.render:
                    self.env.render()
                received_action = self.heuristic_control(state, x)
                next_state, reward, done, _, info = self.env.step(received_action)
                rewards_for_episode.append( reward )
                state = next_state
                if done:
                    break
                        
            rewards_for_episode = np.array(rewards_for_episode)
            total_rewards.append(np.sum(rewards_for_episode))
        total_rewards = np.array(total_rewards)

        # This value is negated by default
        mean_rewards = np.mean(total_rewards)
        return mean_rewards * (-1 if self.negate else 1)