
#   WSI - ćwiczenie 1
#   Prowadzący: mgr inż. Mikołaj Markiewicz
#   Wykonał: Jan Kaniuka
#   Numer indeksu: 303762

import numpy as np
from matplotlib import pyplot as plt


# ------------------------------------------
# Parametry do eksperymentów

start_point = 5
learning_rate = 0.1
error = 10 ** -3
# function type
# - "one_minimum" -> funkcja z jednym minimum
# - "local_minimum" -> funkcja z minimum lokalnym
function_type = "one_minimum"

# -------------------------------------------


# Lista na punkty do wykresu
plot_list=[]


# Przykłady funkcji
if function_type == "local_minimum":
  function = lambda x: x ** 4 - 5*x ** 2 - 3*x
  derivative = lambda gx: 4*gx ** 3 - 10*gx - 3
elif function_type == "one_minimum":
  function = lambda x: x ** 2 + 3*x + 8
  derivative = lambda gx: 2 * gx + 3



def add_points_to_plot(arg, val):
  plot_list.append((arg, val))
  return None


# Wylicznie x(i+1) wg. algorytmu
def calculate_next_x(point):   
  gradient_val = derivative(point)
  d = - gradient_val
  next_x = point + learning_rate * d
  return next_x


def plot_results(function_type, iterations):

  if function_type == "one_minimum":
    x = np.arange(-5,7.5,0.1)
    function_formula = x ** 2 + 3*x + 8

  elif function_type == "local_minimum":
    x = np.arange(-2,3,0.1)
    function_formula = x ** 4 - 5*x ** 2 - 3*x

  plt.plot(x, function_formula)
  x_val = [x[0] for x in plot_list]
  y_val = [y[1] for y in plot_list]
  plt.plot(x_val, y_val, '.r-')
  plt.grid()
  plt.title("Learning rate = {rate} with {num} iterations".format(rate=str(learning_rate).replace(".",","), num=iterations))
  plt.show()

  return None


def gradient_descent(start_point, learning_rate, error):

  iterations = 0


  next_x = calculate_next_x(start_point) 
  previous_x = start_point
  add_points_to_plot(start_point,function(start_point))
  iterations += 1

 # Kryterium stopu
  while ( abs(function(next_x) - function(previous_x)) > error): 

      previous_x = next_x
      next_x = calculate_next_x(previous_x) 
      add_points_to_plot(next_x, function(next_x))
      iterations += 1

  
  return next_x, iterations


if __name__ == "__main__":

  result = gradient_descent(start_point, learning_rate, error)

  print("Minimum at point x =" , result[0])
  print("Reached within", result[1], "iterations")

  plot_results(function_type, result[1])

