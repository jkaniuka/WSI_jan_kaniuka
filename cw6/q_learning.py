import sys, os
import numpy as np
from matplotlib import pyplot as plt
import copy
from numpy import random


# ----------------------- PARAMETERS ---------------------------

filename = 'maze2.txt'

epsilon = 0.001
discount_factor = 0.9
learn_rate = 0.8
epochs = 1000

# --------------------------------------------------------------

# Global variables and lists to collect data
rewards_hist = []
steps_num_hist = []

global q_table
global rewards
global maze_rows
global maze_columns
global start_row
global start_col


moves = ['^', '>', 'v', '<']

# Epsilon-Greedy strategy
def choose_move(row, col, eps):
  if np.random.uniform(0,1) < eps:
    return np.random.randint(4)
  return np.argmax(q_table[row, col])


# randomly choose starting point at the beginning of the epoch
def choose_start_point():
  row, col  = random.randint(maze_rows), random.randint(maze_columns)
  while is_terminal(row, col):
    row, col  = random.randint(maze_rows), random.randint(maze_columns)
  return row, col

# Checking if wall was hit
def is_terminal(row, col):
  if rewards[row, col] == -1:
    return False
  return True

# Maze conversion from .txt format
def convert_maze():
  global q_table
  global rewards
  global maze_rows
  global maze_columns
  global start_row
  global start_col

  all_lines = []

  with open(os.path.join(sys.path[0], filename), 'r') as my_file:
      for line in my_file:
          all_lines.append(line.rstrip("\n"))
        
  all_lines.pop()

  maze_rows = len(all_lines)
  maze_columns = len(all_lines[0])

  for i in range(len(all_lines)):
    while len(all_lines[i]) != maze_columns:
      all_lines[i] += '.'

  q_table = np.zeros((maze_rows, maze_columns, 4))
  rewards = np.full((maze_rows, maze_columns), -100)

  for i in range(maze_rows):
    for j in range(maze_columns):
      if all_lines[i][j] == 'F':
        rewards[i,j] = 100
      if all_lines[i][j] == '.' or all_lines[i][j] == 'S':
          rewards[i, j] = -1
          if all_lines[i][j] == 'S':
            start_row = i 
            start_col = j 


# update on agent's position in labitynth
def get_new_state(row_now, col_now, direction):
  row_next, col_next = row_now, col_now
  if moves[direction] == '^' and row_now > 0:
    row_next -= 1
  elif moves[direction] == '>' and col_now < maze_columns - 1:
    col_next += 1
  elif moves[direction] == 'v' and row_now < maze_rows - 1:
    row_next += 1
  elif moves[direction] == '<' and col_now > 0:
    col_next -= 1
  return row_next, col_next


# return path in the labitynth based on determined policy
def find_path(row_start, col_start):
  row, col = row_start, col_start
  path = []
  path.append([row, col])
  while not is_terminal(row, col):
    row, col = get_new_state(row, col, np.argmax(q_table[row, col]))
    path.append([row, col])
  return path

# moving average filter for data analysis
def MA_filter(data, window_size):
  i = 0
  moving_averages = []
  while i < len(data) - window_size + 1:
      window = data[i : i + window_size]
      window_average = round(sum(window) / window_size, 2)
      moving_averages.append(window_average)
      i += 1  
  return moving_averages

# agent discovers the environment for a given number of epochs
def train_agent():
  for episode in range(epochs):
    gain = 0
    steps = 0

    row, col = choose_start_point()
    while not is_terminal(row, col):
      move = choose_move(row, col, epsilon)
      last_row, last_col = row, col 
      row, col = get_new_state(row, col, move)
      reward = rewards[row, col]
      # gain counter
      gain = gain + reward
      new_q_value = q_table[last_row, last_col, move] + learn_rate * (reward + (discount_factor * np.max(q_table[row, col])) - q_table[last_row, last_col, move])
      q_table[last_row, last_col, move] = new_q_value
      # steps counter
      steps = steps + 1
    
    # history update
    rewards_hist.append(gain)
    steps_num_hist.append(steps)


def plot_stats():
  fig = plt.figure(figsize=(10, 6))

  ax = fig.add_subplot(121)
  ax2 = fig.add_subplot(122)

  
  ax.plot(MA_filter(rewards_hist,80))
  ax.set_xlabel('epochs')
  ax.set_ylabel('reward per epoch')

  ax2.plot(MA_filter(steps_num_hist,5))
  ax2.set_xlabel('epochs')
  ax2.set_ylabel('steps per epoch')

# plot solution (labirynth)
def plot_maze(row, col):
  global maze_columns
  visualization = copy.copy(rewards)
  path_to_goal = find_path(row, col)
  print(path_to_goal)
  for element in path_to_goal:
    visualization[element[0],element[1]] = 99
  plt.figure(figsize=(5,5))
  plt.title("Maze solution")
  plt.axis('off')
  plt.imshow(visualization)

def show_plots():
  plt.show()


if __name__ == "__main__":
  convert_maze()
  train_agent()
  plot_stats()
  plot_maze(start_row, start_col)
  show_plots()

