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
    1: {"name": "5-long", "arr": np.array([[1, 1, 1, 1, 1]], dtype=int)},
    2: {"name": "4-long", "arr": np.array([[1, 1, 1, 1]], dtype=int)},
    3: {"name": "3-long", "arr": np.array([[1, 1, 1]], dtype=int)},
    4: {"name": "2-long", "arr": np.array([[1, 1]], dtype=int)},
    5: {"name": "1-dot", "arr": np.array([[1]], dtype=int)},
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
        self.grid_space = spaces.Box(low=0, high=1, shape=self.grid_shape, dtype=int)
        # pieces_space is each of the 3 piece elements
        self.pieces_space = spaces.MultiDiscrete([20, 20, 20])
        
        # Observation space includes the grid AND pieces we have available
        self.observation_space = spaces.Tuple((self.grid_space, self.pieces_space))
        
        # Action space accounts for all the possible placements of 3 pieces
        self.action_space = spaces.Discrete(300)
        
        # numpy representation of the grid
        self.grid = np.zeros(self.grid_shape, dtype=int)
        
        # Grid with colors added to represent different blocks
        self.color_grid = np.zeros((self.grid_shape[0], self.grid_shape[1], 3),
                                   dtype=int)
        
        # pieces available
        self.pieces = [0, 0, 0]
        
        # Start the game off with 0 points
        self.points = 0
    
    def pick_new_pieces(self, choices=[None, None, None]):
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
        self.grid = np.zeros_like(self.grid)
        
    def expand_arr(self, arr, mult=2, target_shape=(10,10)):
        new_arr = np.zeros_like(arr, shape=(arr.shape[0] * mult, arr.shape[1]))
        for idx, y in enumerate(arr):
            for m in range(mult):
                new_arr[idx * mult + m] = y
        
        newer_arr = np.zeros_like(new_arr, shape=(new_arr.shape[0], new_arr.shape[1] * mult))
        for idx, x in enumerate(new_arr.T):
            for m in range(mult):
                newer_arr[:, idx * mult + m] = x.T
            
        final_arr = np.zeros_like(newer_arr, shape=target_shape)
        
        start_y = int((target_shape[0] - newer_arr.shape[0]) / mult)
        start_x = int((target_shape[1] - newer_arr.shape[1]) / mult)
        
        final_arr[start_y:start_y+newer_arr.shape[0], start_x:start_x+newer_arr.shape[1]] = newer_arr
        
        return final_arr
        
    def render(self, mode ="human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        
        expand_grid = self.expand_arr(np.copy(self.grid), 
                                      mult=3, 
                                      target_shape=(self.grid_shape[0] * 3,
                                                    self.grid_shape[1] * 3))
        exp_pieces = []
        for i in self.pieces:
            if i == 0:
                exp_pieces.append(np.zeros((10, 10), dtype=int))
            else:
                exp_pieces.append(self.expand_arr(np.copy(pieces_specs[i]["arr"])))
            
        pieces_arr = np.concatenate(exp_pieces, axis=1)
        
        final_show = np.concatenate((expand_grid, 
                                     np.ones((5, expand_grid.shape[1])) * 0.5,
                                     pieces_arr))
        if mode == "human":
            cv2.namedWindow("1010!", cv2.WINDOW_NORMAL)
            cv2.imshow("1010!", final_show)
            cv2.resizeWindow('1010!', final_show.shape[1] * 20,final_show.shape[0] * 20)
            cv2.waitKey(0)
        
        elif mode == "rgb_array":
            return final_show
        
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
        
    
        # Add the block to the grid
        # valid is true if the move was valid, false if not
        # reward is the additional points from executing that move
        done, reward = self.execute_action(action)
    
        return (self.grid, self.pieces), reward, done, []
    
    def row_col_clear_points(self, num):
        if num < 1:
            return 0
        elif num == 1:
            return 10
        else:
            return (10 * num) + self.row_col_clear_points(num - 1)
    
    def check_valid_move(self, action, ret_grid=True):
        ''' 
        Returns:
            done: Boolean - whether the action ends the game or not
            grid (optional): ndarray - returns the grid with piece added
        '''
        # Get which piece is being added to the grid
        piece_idx = int(np.floor(action/100))
        
        # Check if trying to act on a null piece, return if so
        if self.pieces[piece_idx] == 0:
            if ret_grid:
                return True, self.grid, 0
            else:
                return True
        
        # Get the array of the action piece
        piece_arr = pieces_specs[self.pieces[piece_idx]]["arr"]
        piece_shape = piece_arr.shape
        
        # action index, independent of piece
        relative_action = action - (piece_idx * 100)
        
        # Position of action on the grid
        grid_pos_y = int(np.floor(relative_action / 10))
        grid_pos_x = relative_action % 10
        
        
        # Check if the piece is out of bounds, return if it is
        if (grid_pos_y + piece_shape[0]) > self.grid_shape[0] or\
        (grid_pos_x + piece_shape[1]) > self.grid_shape[1]:
            if ret_grid:
                return True, self.grid, 0
            else:
                return True
        
        # Check if there are any overlaps, return if so
        if np.max(self.grid[grid_pos_y:grid_pos_y+piece_shape[0], 
                   grid_pos_x:grid_pos_x+piece_shape[1]]) > 0:
            if ret_grid:
                return True, self.grid, 0
            else:
                return True
        elif not ret_grid:
            return False
        
        # Superimpose the piece onto the grid and return
        grid_manip = np.copy(self.grid)
        grid_manip[grid_pos_y:grid_pos_y+piece_shape[0], 
                   grid_pos_x:grid_pos_x+piece_shape[1]] += piece_arr
        
        # Set that piece index to 0
        self.pieces[piece_idx] = 0
        
        # Reward for placing the piece
        reward = np.sum(piece_arr)
        
        return False, grid_manip, reward
    
    
    def execute_action(self, action):
        done, grid_manip, reward = self.check_valid_move(action)
        
        if done:
            return True, reward
        
        # Reward for any rows/columns completed
        rows = []
        cols = []
        for y in range(self.grid_shape[0]):
            if np.sum(grid_manip[y]) == self.grid_shape[0]:
                rows.append(y)
        for x in range(self.grid_shape[1]):
            if np.sum(grid_manip[:, x]) == self.grid_shape[1]:
                cols.append(x)
        
        # Update the reward/points total
        reward += self.row_col_clear_points(len(rows) + len(cols))
        self.points += reward
        
        self.grid = grid_manip
        
        # Actually clear the rows from the grid
        for row in rows:
            self.grid[row] = np.zeros_like(self.grid[row])
        for col in cols:
            self.grid[:, col] = np.zeros_like(self.grid[:, col])
        
        # Reset the pieces if we just ran out
        if np.sum(self.pieces) == 0:
            self.pick_new_pieces()
        
        # Check if there are no more moves
        for i in range(self.action_space.n):
            done = self.check_valid_move(i, ret_grid=False)
            if not done:
                break
        
        if done == True:
            print("done is true")
        
        # Returns done as false if there are remaining moves available
        return done, reward
    

if __name__ == "__main__":
#    from IPython import display
    
    env = game_1010()
    obs = env.reset()
    
    
    while True:
        done = True
        reward = 0
        # Take a random action
        while done == True and reward == 0:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(action)
        
        # Render the game
        env.render()
        
        if done == True:
            break
    print("HERE")
    env.render()
    env.close()