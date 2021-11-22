#   WSI - ćwiczenie 3
#   Prowadzący: mgr inż. Mikołaj Markiewicz
#   Wykonał: Jan Kaniuka
#   Numer indeksu: 303762


from random import choice
import numpy as np

# initial empty board
board = [[' ', ' ', ' '], [' ', ' ', ' '],[' ', ' ', ' ']]

# winning configurations
def winner_table(board):
  winner_lookup_table = [[board[0][0], board[0][1], board[0][2]], # ROWS
                  [board[1][0], board[1][1], board[1][2]],
                  [board[2][0], board[2][1], board[2][2]],

                  [board[0][0], board[1][0], board[2][0]], # COLUMNS
                  [board[0][1], board[1][1], board[2][1]],
                  [board[0][2], board[1][2], board[2][2]],

                  [board[0][0], board[1][1], board[2][2]], # DIAGONALS
                  [board[0][2], board[1][1], board[2][0]]]
  return winner_lookup_table

# Chars for printing purposes
playerX = 'X'
playerO = 'O'
blank_cell = ' '

# ************************** START OF PARAMETERS ******************************
# Artificial inteligence 1 (CPU1)
depth1 = 7
ab_prun1 = False
global counter1 # moves in game
global sum1 # sum of moves
sum1 = 0
counter1 = 0


# Artificial inteligence 2 (CPU2)
depth2 = 2
ab_prun2 = True
global counter2 # moves in game
global sum2 # sum of moves
sum2 = 0
counter2 = 0

# Number  of iterations
iterations = 10

# Random begin of the game (only first, start move is selected randomly )
first_random = True

# AI1 - random game mode
full_random = False



# ************************** END OF PARAMETERS ******************************

# Statistics
global x_win
x_win = 0
global tie 
tie = 0


# Print board in console
def show_board(board):
    for row in board:
      for cell in row:
        print(cell,'|',end='')
      print('\n'+ '---------')
    print('\n')

# Check for possible winner-combination
def get_winner(board, token):
    if [token, token, token] in winner_table(board):
        return True

    return False

# Check if we have a winner ( O or X)
def end_of_game(board):
  if get_winner(board, playerX):
    return get_winner(board, playerX)
  elif get_winner(board, playerO):
    return get_winner(board, playerO)

# Clear board before new game
def erase(board):
    for x in range(3):
        for y in range(3):
            board[x][y] = ' '

# Get statistics for report purposes
def stats_update(board):
    global x_win
    global tie
    if get_winner(board, playerX):
      x_win += 1
    elif get_winner(board, playerO):
      pass
    else:
        tie += 1 

# Board is full, without winner (it's draw)
def draw(board):
    if len(available_cells(board)) == 0:
        return True
    return False

# Makes list of possible moves on board
def available_cells(board):
  available = []
  for i in range(3):
      for j in range(3):
          if board[i][j] == ' ':
              available.append([i, j])

  return available

# Labeling the node with +1/0/-1 (no heuristic function)
def node_result(board):
    if get_winner(board, playerX):
        return 1
    elif get_winner(board, playerO):
        return -1
    else:
        return 0

# Adding a sign/token to the board
def put_char(board, x, y, player):
    board[x][y] = player

# Minimax algorithm with alpha-beta pruning
def minimax(board, depth, alpha, beta, player, num):
    global counter1
    global counter2
    row = -1
    col = -1
    if depth == 0 or end_of_game(board):
        return [row, col, node_result(board)]
    else:
        for cell in available_cells(board):
          # Game stats
            if num == 1:
              counter1 +=1
            else:
              counter2 +=1
            # Simulate move
            put_char(board, cell[0], cell[1], player)
            evaluation = minimax(board, depth - 1, alpha, beta, switch_player(player), num)
            if player == playerX:
                # X maximizes
                if evaluation[2] > alpha:
                    alpha = evaluation[2]
                    row = cell[0]
                    col = cell[1]
            else:
                if evaluation[2] < beta:
                    beta = evaluation[2]
                    row = cell[0]
                    col = cell[1]
            # reverse simulated move -> put empty cell 
            put_char(board, cell[0], cell[1], blank_cell)
            # alpha-beta pruning - pruning condition
            if alpha >= beta and ((num == 1 and ab_prun1 == True) or (num == 2 and ab_prun2 == True)):
              break

        if player == playerX:
            return [row, col, alpha] # alpha because X is maximizing
        else:
            return [row, col, beta]


# AI1 player (with full-random game option)
def CPU1(board):
  if full_random:
    cell_found = False
    while not cell_found:
        random_row = choice([0, 1, 2])
        random_column = choice([0, 1, 2])
        put_char(board, random_row, random_column, playerX)
        show_board(board)
        cell_found = True
  else:
    # Radom place at the beginning of the game
    if len(available_cells(board)) == 9 and first_random:
        random_row = choice([0, 1, 2])
        random_column = choice([0, 1, 2])
        put_char(board, random_row, random_column, playerX)
        show_board(board)
    else:
        best_position = minimax(board, depth1, -np.inf, np.inf, playerX, 1)
        put_char(board, best_position[0], best_position[1], playerX)
        show_board(board)

# AI2 player
def CPU2(board):
  best_position = minimax(board,depth2, -np.inf, np.inf, playerO, 2)
  put_char(board, best_position[0], best_position[1], playerO)
  show_board(board)

# opponents move
def next_turn(board, player, mode):
    if mode == 1:
      pass
    else:
        if player == playerX:
            CPU1(board)
        else:
          CPU2(board)

# switches from min to max between layers in game tree
def switch_player(player):
  if player == 'X':
    return 'O'
  elif player == 'O':
   return 'X'

# play CPU1 vs CPU2
def CPU1vsCPU2():
    whos_turn = playerX
    erase(board)
    while not (end_of_game(board) or draw(board)):
        next_turn(board, whos_turn, 2)
        whos_turn = switch_player(whos_turn)

# Game loop
def main():
    global sum1
    global sum2
    global counter1
    global counter2
    for x in range(iterations):
      CPU1vsCPU2()
      stats_update(board)
      sum1 += counter1
      sum2 += counter2
      counter1 = 0
      counter2 = 0

    print('X przeszukane stany: ', sum1, 'O przeszukane stany: ', sum2)
    print('Wygrane/przegrane/remis gracza 1: ')
    print(x_win, '/',iterations - x_win-tie ,'/',tie)



if __name__ == '__main__':
    main()