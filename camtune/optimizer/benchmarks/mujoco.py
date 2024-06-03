import torch
import numpy as np
import gym
from botorch.test_functions import Ackley

from .base_benchmark import BaseBenchmark

from camtune.utils.logger import print_log
from camtune.utils.vars import DEVICE, DTYPE

HOPPER_MEAN = np.array([1.41599384, -0.05478602, -0.25522216, -0.25404721, 
                        0.27525085, 2.60889529,  -0.0085352, 0.0068375, 
                        -0.07123674, -0.05044839, -0.45569644])
HOPPER_STD = np.array(
                [0.19805723, 0.07824488, 0.17120271, 0.32000514, 
                 0.62401884, 0.82814161, 1.51915814, 1.17378372, 
                 1.87761249, 3.63482761, 5.7164752])

class Swimmer(BaseBenchmark):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.negate = False

        self.env = gym.make('Swimmer-v2')
        self.render = False
        self.policy_shape = (2, 8)
        self.num_rollouts = 3

        self.mean, self.std = 0, 1
        self.counter      = 0
        
        # tunable hyper-parameters in LA-MCTS
        self.mcts_params = {
            "Cp": 20,
            "leaf_size": 10,
            "global_num_init": 40,
            "classifier_params": {
                "kernel_type": "poly",
                "gamma_type": "scale",
            }
        }

        self.__post_init__()
        
        
    def __call__(self, x: torch.Tensor):
        x = x.detach().cpu().numpy()
        self.counter += 1
        
        M = x.reshape(self.policy_shape)
        
        returns = []
        observations = []
        actions = []
        
        
        for i in range(self.num_rollouts):
            obs    = self.env.reset()
            done   = False
            totalr = 0.
            steps  = 0
            
            while not done:
                
                action = np.dot(M, (obs - self.mean)/self.std)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps += 1                
                if self.render:
                    self.env.render()            
            returns.append(totalr)
            
        
        return np.mean(returns) * (-1 if self.negate else 1)
    
class Hopper(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.counter = 0
        self.env     = gym.make('Hopper-v4')
        self.num_rollouts = 3
        self.render  = False
        self.policy_shape = (3, 11)
        
        self.mcts_params = {
            "Cp": 10,
            "leaf_size": 100,
            "global_num_init": 150,
            "classifier_params": {
                "kernel_type": "poly",
                "gamma_type": "auto",
            }
        }
        self._obj_func = self
        
        self.__post_init__()
            
    def __call__(self, x: torch.Tensor):
        self.counter += 1
        x = x.detach().cpu().numpy()
        M = x.reshape(self.policy_shape)
        
        returns = []
        observations = []
        actions = []
        
        for i in range(self.num_rollouts):
            obs    = self.env.reset()
            done   = False
            totalr = 0.
            steps  = 0
            while not done:
                # M      = self.policy
                inputs = (obs - HOPPER_MEAN) / HOPPER_STD
                action = np.dot(M, inputs)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps  += 1
                if self.render:
                    self.env.render()
            returns.append(totalr)
            
        return np.mean(returns) * (-1 if self.negate else 1)