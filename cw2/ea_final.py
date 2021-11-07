import random
from numpy.core.fromnumeric import size, var 
from matplotlib import pyplot as plt
import numpy as np
from math import sin, cos, exp 
import time

#  start of program execution
start_time = time.time()

# *************************** EXAMPLES OF FUNCTIONS ***************************
# --> x is a tuple containing funtion arguments

# -------------------  2D-functions -------------------

#function = lambda x: x[0] ** 2 

#function = lambda x: x[0] ** 4 - 5*x[0] ** 2 - 3*x[0]

# -------------------  3D-functions -------------------

# Paraboloid
#function = lambda x : x[0]**2 + x[1]**2

# Rosenbrock function
function = lambda x : (1-x[0])**2+100*(x[1]-x[0]**2)**2

# Bird function
#function = lambda x : sin(x[0])*exp(1-cos(x[1]))**2 + cos(x[1])*exp(1-sin(x[0]))**2+(x[0]-x[1])**2

# Shubert Function
# def function(x):
#       sum1=0
#       sum2=0
#       for i in range(1,6):
#           sum1 = sum1 + (i* cos(((i+1)*x[0]) +i))
#           sum2 = sum2 + (i* cos(((i+1)*x[1]) +i))
#       return sum1 * sum2



# *******************************  Parameters  **************************************

var_num = 2 # dimensionality of the problem (number of variables)
population_size = 200
population_type = "random" # "random" or "clone" as instructed

# range for gene randomization in the chromosome
upper_bound = 5
lower_bound = -5

pm = 0.2 # probability of mutation
mut_coeff = 0.55 # mutation strength (sigma)
elite_size = 1
tournament_size = 3
alfa = 0.1 # coefficient for arithmetic recombination
iterations = 500

# *******************************  End of Parameters  *******************************

# iterator for while loop
global t
t = 0 

# coordinates and value for the best point IN THE CURRENT ITERATION
global optimal_point
optimal_point = []
global optimal_val
optimal_val = 0

# coordinates and value for the best point SO FAR
global best_point_now
best_point_now = []
global best_val_now
best_val_now = 0



def initialize_population(type): 
  population = []
  local_list = []
  if type == "clone":
      seed = random.uniform(lower_bound, upper_bound)
      for j in range(var_num):
        local_list.append(seed)
      for i in range (population_size):
          population.append(local_list)
  elif type == "random":
    for i in range (population_size) :
      for j in range(var_num):
        seed = random.uniform(lower_bound, upper_bound)
        local_list.append(seed)
      population.append(local_list)
      local_list = []
  else:
    raise Exception('Wrong type of population!')
  return population

# calculate value at given point
def evaluate(args):
  return function(tuple(args))


# return best individual
def find_best(population):
  candidates = []
  for i in range(population_size):
          candidates.append([population[i], evaluate(population[i])])
  return sorted(candidates, key=lambda y: y[1], reverse=False)

# order individuals from the best
def sort_population(population):
  return sorted(population, key=lambda y: y[1], reverse=False)


def tournament_selection(evaluated_list):
  selected = []
  fight_list = []
  best_list = []
  for i in range(population_size):
    for j in range(tournament_size):
      fight_list.append(evaluated_list[random.randrange(0,population_size)])
    best_list = sort_population(fight_list)
    fight_list = []
    selected.append(best_list[0])
    
  return selected

#  full arithmetic recombination
def crossover(reproducted):
  crossed = []
  for i in range(population_size):
    parent_1 = reproducted[random.randrange(0,population_size)]
    parent_2 = reproducted[random.randrange(0,population_size)]

    # child 1
    child_1 = []
    child_1_point = []
    for i in range(var_num):
      child_1_point.append(alfa * parent_1[0][i] + (1 - alfa) * parent_2[0][i])
    child_1.append(child_1_point)
    child_1.append(evaluate(child_1_point))

    # child 2
    child_2 = []
    child_2_point = []
    for i in range(var_num):
      child_2_point.append(alfa * parent_2[0][i] + (1 - alfa) * parent_1[0][i])
    child_2.append(child_2_point)
    child_2.append(evaluate(child_2_point))

    # population size must be preserved! 
    if len(crossed) < population_size:
      crossed.append(child_1)
    if len(crossed) < population_size:
      crossed.append(child_2)

  return crossed 



def mutation(before_mutation):
  mutated = before_mutation
  new_list = []
  for i in range(population_size):
    buffer = []
    for j in range(var_num):
      rand_val = random.uniform(0,1)
      if rand_val < pm:
        modified = mutated[i][0][j] + mut_coeff * random.gauss(0,1)
        buffer.append(modified)
      else:
        buffer.append(mutated[i][0][j])
    new_list.append([buffer, evaluate(buffer)])

  return new_list 



# the new generation is the elite of the current generation and the best individuals after the mutation 
def succession(population, after_mutation, elite_size):
  if elite_size > population_size:
    raise Exception('Elite size is greater than population size!')
  next_generation = []
  for i in range(elite_size):
    next_generation.append(sorted(population, key=lambda y: y[1], reverse=False)[i])
  for j in range(population_size - elite_size):
    next_generation.append(sorted(after_mutation, key=lambda y: y[1], reverse=False)[j])
  return next_generation

def main():
  global optimal_point
  global optimal_val
  global best_point_now
  global best_val_now

  global t

  population = initialize_population(population_type)

  sorted_population= find_best(population)

  # Optimal point IN CURRENT ITERATION
  optimal_point = sorted_population[0][0]
  optimal_val = sorted_population[0][1]

  while t < iterations:

    after_tournament = tournament_selection(sorted_population)

    after_crossover = crossover(after_tournament)

    after_mutation = mutation(after_crossover)

    # Find best individual IN CURRENT ITERATION
    sorted_after_mut = sort_population(after_mutation)
    [best_point_now, best_val_now]  = sorted_after_mut[0]

    # Checking if a better individual was found
    if best_val_now < optimal_val:
      optimal_val = best_val_now
      optimal_point = best_point_now


    # succession
    sorted_population = succession( sorted_population, after_mutation, elite_size)

    t += 1

  # print solution 
  print("Function minimum at: ", optimal_point, "with value of: ", optimal_val)


if __name__ == '__main__':
    main()


#  end of program execution
print("--- %s seconds ---" % (time.time() - start_time))