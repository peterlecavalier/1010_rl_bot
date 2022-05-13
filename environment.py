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


class game_1010_v0(Env):
    def __init__(self, seed=None):
        super(game_1010_v0, self).__init__()

        # Grid observation space
        self.grid_shape = (10, 10)
        # grid_space is the 10x10 grid
        self.grid_space = spaces.Box(low=0, high=1, shape=self.grid_shape, dtype=int)
        # pieces_space is each of the 3 piece elements
        self.pieces_space = spaces.Box(low=0, high=20, shape=(3,), dtype=int)
        self.valid_mask_space = spaces.Box(low=0, high=1, shape=(300,), dtype=int)

        # Observation space includes the grid, pieces we have available, and invalid action mask
        self.state_space = spaces.Tuple((self.grid_space, self.pieces_space))
        self.observation_space = spaces.Dict({
            "action_mask": self.valid_mask_space,
            "state": self.state_space
        })
        
        # Action space accounts for all the possible placements of 3 pieces
        self.action_space = spaces.Discrete(300)
        
        # numpy representation of the grid
        self.grid = np.zeros(self.grid_shape, dtype=int)
        
        # Grid with colors added to represent different blocks
        self.color_grid = np.zeros((self.grid_shape[0], self.grid_shape[1], 3),
                                   dtype=int)
        
        # pieces available
        self.pieces = [0, 0, 0]
        self.pieces_colors = [None, None, None]
        
        # Valid moves
        self.valid_moves = [0] * 300
        
        # Start the game off with 0 points
        self.points = 0
        
        # Set up rng
        # if seed is not None:
        #     self.rng, self.rng_seed = gym.utils.seeding.np_random(seed)
        # else:
        self.rng = np.random
        self.rng_seed = 0

    def reset(self):
        # Reset points back to 0
        self.points = 0
        
        # Generate random pieces
        self.pick_new_pieces()

        # All moves are valid
        self.valid_moves = [1] * 300
        
        # Reset grid to be zero everywhere
        self.grid = np.zeros_like(self.grid)
        self.color_grid = np.zeros_like(self.color_grid)

        return {"action_mask": self.valid_moves, "state": (self.grid, self.pieces)}

    def step(self, action):
        # Terminate an episode if:
        # 1) we lose the game
        # 2) the agent attempts to use an empty piece
        # 3) the agent attempts to place a piece in an invalid spot
        
        # The action must be in the bounds of the action space first
        assert self.action_space.contains(action), "Invalid Action"
        
        # Add the block to the grid
        # valid is true if the move was valid, false if not
        # reward is the additional points from executing that move
        done, reward = self.execute_action(action)
    
        return {"action_mask": self.valid_moves, "state": (self.grid, self.pieces)}, reward, done, {}

    def pick_new_pieces(self, choices=[None, None, None]):
        # Pick new pieces for the game
        # Also handles invalid action masking
        for i in range(3):
            if choices[i] is None:
                self.pieces[i] = self.rng.randint(1, 20)
            else:
                self.pieces[i] = choices[i]
            self.pieces_colors[i] = self.rng.randint(256, size=3)

    def update_valid_moves(self):
        # Check which moves are valid
        for i in range(self.action_space.n):
            done = self.check_valid_move(i, ret_grid=False)
            if not done:
                self.valid_moves[i] = 1
            else:
                self.valid_moves[i] = 0  
        
    def expand_arr(self, arr, mult=2, target_shape=(10,10, 3)):
        new_arr = np.zeros_like(arr, shape=(arr.shape[0] * mult, arr.shape[1], 3))
        for idx, y in enumerate(arr):
            for m in range(mult):
                new_arr[idx * mult + m] = y
        
        newer_arr = np.zeros_like(new_arr, shape=(new_arr.shape[0], new_arr.shape[1] * mult, 3))
        for idx, x in enumerate(np.transpose(new_arr, axes=(1, 0, 2))):
            for m in range(mult):
                newer_arr[:, idx * mult + m] = x
            
        final_arr = np.zeros_like(newer_arr, shape=target_shape)
        
        start_y = int((target_shape[0] - newer_arr.shape[0]) / mult)
        start_x = int((target_shape[1] - newer_arr.shape[1]) / mult)
        
        final_arr[start_y:start_y+newer_arr.shape[0], start_x:start_x+newer_arr.shape[1]] = newer_arr
        
        return final_arr
    
    def colorize_piece(self, piece_idx):
        # Generate a random color for the piece and apply it to the array
        copy_arr = np.copy(pieces_specs[self.pieces[piece_idx]]["arr"])
        copy_arr = np.stack((copy_arr, copy_arr, copy_arr), axis=-1)

        rgb = self.pieces_colors[piece_idx]
        copy_arr = np.multiply(copy_arr, rgb)
        
        return copy_arr
        
    def render(self, mode ="evaluate"):
        assert mode in ["human", "rgb", "evaluate"], "Invalid mode, must be either \"human\", \"evaluate\", or \"rgb_array\""
        
        expand_grid = self.expand_arr(np.copy(self.color_grid), 
                                      mult=3, 
                                      target_shape=(self.grid_shape[0] * 3,
                                                    self.grid_shape[1] * 3, 3))
        exp_pieces = []
        for idx, i in enumerate(self.pieces):
            if i == 0:
                exp_pieces.append(np.zeros((10, 10, 3), dtype=int))
            else:
                exp_pieces.append(self.expand_arr(self.colorize_piece(idx)))
            
        pieces_arr = np.concatenate(exp_pieces, axis=1)
        
        final_show = np.concatenate((expand_grid, 
                                     np.ones((5, expand_grid.shape[1], 3)) * 255,
                                     pieces_arr)).astype(np.uint8)
        
        final_show = cv2.resize(final_show, (final_show.shape[1] * 20,
                                             final_show.shape[0] * 20),
                                interpolation=cv2.INTER_NEAREST)
        final_show = cv2.putText(final_show, f"Points: {self.points}",
                                 (50, 660), font, 2,
                                 (0, 0, 0), 2, cv2.LINE_AA)
        if mode == "human":
            cv2.namedWindow("1010!", cv2.WINDOW_NORMAL)
            cv2.imshow("1010!", final_show)
            cv2.waitKey(0)
        if mode == "evaluate":
            cv2.namedWindow("1010!", cv2.WINDOW_NORMAL)
            cv2.imshow("1010!", final_show)
            cv2.waitKey(5)
        
        elif mode == "rgb":
            return final_show
        
    def close(self):
        cv2.destroyAllWindows()
    
    def row_col_clear_points(self, num):
        if num < 1:
            return 0
        elif num == 1:
            return 10
        else:
            return (10 * num) + self.row_col_clear_points(num - 1)
    
    def check_valid_move(self, action, ret_grid=True, bad_reward=-50):
        ''' 
        Returns:
            done: Boolean - whether the action ends the game or not
            grid (optional): ndarray - returns the grid with piece added
            color_grid (optional): ndarray - returns the color grid with piece added
            reward (optional): int - reward from the move
        '''
        # Get which piece is being added to the grid
        piece_idx = int(np.floor(action/100))
        
        # Check if trying to act on a null piece, return if so
        if self.pieces[piece_idx] == 0:
            if ret_grid:
                return True, self.grid, self.color_grid, bad_reward#-self.points - 1
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
                return True, self.grid, self.color_grid, bad_reward#-self.points - 1
            else:
                return True
        
        
        
        # Superimpose the piece onto the grid
        grid_manip = np.copy(self.grid)
        grid_manip[grid_pos_y:grid_pos_y+piece_shape[0], 
                   grid_pos_x:grid_pos_x+piece_shape[1]] += piece_arr
        
        # Check if there are any overlaps, return if so
        if np.max(grid_manip) > 1:
            if ret_grid:
                return True, self.grid, self.color_grid, bad_reward#-self.points - 1
            else:
                return True
        elif not ret_grid:
            return False
        
        # Add the piece to the color grid
        color_grid_manip = np.copy(self.color_grid)
        color_grid_manip[grid_pos_y:grid_pos_y+piece_shape[0], 
                   grid_pos_x:grid_pos_x+piece_shape[1]] += self.colorize_piece(piece_idx)
        
        # Set that piece index to 0
        self.pieces[piece_idx] = 0
        
        # Reward for placing the piece
        reward = np.sum(piece_arr)
        
        return False, grid_manip, color_grid_manip, reward
    
    
    def execute_action(self, action):
        done, grid_manip, color_grid_manip, reward = self.check_valid_move(action)
        
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
        self.color_grid = color_grid_manip
        
        # Actually clear the rows from the grid
        for row in rows:
            self.grid[row] = np.zeros_like(self.grid[row])
            self.color_grid[row] = np.zeros_like(self.color_grid[row])
        for col in cols:
            self.grid[:, col] = np.zeros_like(self.grid[:, col])
            self.color_grid[:, col] = np.zeros_like(self.color_grid[:, col])
        
        # Reset the pieces if we just ran out
        if np.max(self.pieces) == 0:
            self.pick_new_pieces()
        
        self.update_valid_moves()

        if np.max(self.valid_moves) == 0:
            done = True
        else:
            done = False
        
        # Returns done as false if there are remaining moves available
        return done, reward

if __name__ == "__main__":   
    # If you want to test on a specific seed
    # env = game_1010_v0(seed=0)
    # env.action_space.seed(0)
    
    env = game_1010_v0()
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
        env.render("human")
        
        if done == True:
            break
    print("HERE")
    # env.render()
    env.close()