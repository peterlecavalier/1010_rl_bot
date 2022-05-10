# -*- coding: utf-8 -*-
"""
Created on Fri May  6 16:54:53 2022

@author: Peter
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
import cv2
import random

from gym import Env, spaces
import time

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 


pieces_specs = {
    1: {"name": "5-long", "arr": np.array([1, 1, 1, 1, 1], dtype=int)},
    2: {"name": "4-long", "arr": np.array([1, 1, 1, 1], dtype=int)},
    3: {"name": "3-long", "arr": np.array([1, 1, 1], dtype=int)},
    4: {"name": "2-long", "arr": np.array([1, 1], dtype=int)},
    5: {"name": "1-dot", "arr": np.array([1], dtype=int)},
    6: {"name": "small-l-ur", "arr": np.array([[1, 1], 
                                               [1, 0]], dtype=int)},
    7: {"name": "small-l-ul", "arr": np.array([[1, 1], 
                                               [0, 1]], dtype=int)},
    8: {"name": "small-l-dr", "arr": np.array([[1, 0], 
                                               [1, 1]], dtype=int)},
    9: {"name": "small-l-dl", "arr": np.array([[0, 1], 
                                               [1, 1]], dtype=int)},
    10: {"name": "5-tall", "arr": np.array([[1], 
                                            [1],
                                            [1],
                                            [1],
                                            [1]], dtype=int)},
    11: {"name": "4-tall", "arr": np.array([[1], 
                                            [1],
                                            [1],
                                            [1]], dtype=int)},
    12: {"name": "3-tall", "arr": np.array([[1], 
                                            [1],
                                            [1]], dtype=int)},
    13: {"name": "2-tall", "arr": np.array([[1], 
                                            [1]], dtype=int)},
    14: {"name": "3-3", "arr": np.array([[1, 1, 1], 
                                         [1, 1, 1],
                                         [1, 1, 1]], dtype=int)},
    15: {"name": "big-l-ul", "arr": np.array([[1, 1, 1], 
                                              [0, 0, 1],
                                              [0, 0, 1]], dtype=int)},
    16: {"name": "big-l-dr", "arr": np.array([[1, 0, 0], 
                                              [1, 0, 0],
                                              [1, 1, 1]], dtype=int)},
    17: {"name": "big-l-ur", "arr": np.array([[1, 1, 1], 
                                              [1, 0, 0],
                                              [1, 0, 0]], dtype=int)},
    18: {"name": "big-l-dl", "arr": np.array([[0, 0, 1], 
                                              [0, 0, 1],
                                              [1, 1, 1]], dtype=int)},
    19: {"name": "2-2", "arr": np.array([[1, 1], 
                                         [1, 1]], dtype=int)}
}


class game_1010(Env):
    def __init__(self):
        super(game_1010, self).__init__()
        
        # Grid observation space
        self.grid_shape = (10, 10)
        # grid_space is the 10x10 grid
        self.grid_space = spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=int)
        # pieces_space is each of the 3 piece elements
        self.pieces_space = spaces.MultiDiscrete([20, 20, 20])
        
        # Observation space includes the grid AND pieces we have available
        self.observation_space = spaces.Tuple(self.grid_space, self.pieces_space)
        
        
        # Action space accounts for all the possible placements of 3 pieces
        self.action_space = spaces.Discrete(300)
        
        # numpy representation of the grid
        self.grid = np.zeros(self.observation_shape, dtype=int)
        
        # pieces available
        self.pieces = [0, 0, 0]
        
        # Start the game off with 0 points
        self.points = 0
    
    def pick_new_pieces(self, choices=[None, None, None]):
        # TODO: Modify this to allow human input
        for i in range(3):
            if choices[i] is None:
                self.pieces[i] = np.random.randint(1, 20)
            else:
                self.pieces[i] = choices[i]
        
    
    def reset(self):
        # Reset points back to 0
        self.points = 0
        
        # Generate random pieces
        self.pick_new_pieces()
        
        # Reset grid to be zero everywhere
        self.grid = np.zeros(self.observation_shape, dtype=int)
        
    def render(self, mode ="human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Game", self.grid)
            cv2.waitKey(10)
        
        elif mode == "rgb_array":
            return self.grid
        
    def close(self):
        cv2.destroyAllWindows()
    
    def step(self, action):
        # Terminate an episode if:
        # 1) we lose the game
        # 2) the agent attempts to use an empty piece
        # 3) the agent attempts to place a piece in an invalid spot
        done = False
        
        # The action must be in the bounds of the action space first
        assert self.action_space.contains(action), "Invalid Action"
        
    
        # add the block to the grid
        # TODO: Write below function
        # valid is true if the move was valid, false if not
        # reward is the additional points from executing that move
        done, valid, reward = self.execute_action(action)
        
        if not valid: done = True
    
        # Draw elements on the canvas
        self.draw_elements_on_canvas()
    
        return (self.grid, self.pieces), reward, done, []
        
        
        

class piece:
    def __init__(self, shape_id=None):
        # Create a random piece
        self.name, self.id, self.array, self.shape = self.generate_piece(shape_id)
        
    def generate_piece(self, shape_id=None):
        # return the name, id, array, and shape
        if shape_id is None:
            name = random.choice(list(pieces_specs.keys()))
        else:
            name = shape_id
            
        shape_dict = pieces_specs[name]
        ret_id = shape_dict['id']
        arr = shape_dict['id']
        shape = arr.shape
        
        return name, ret_id, arr, shape
        