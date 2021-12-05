#   WSI - ćwiczenie 4
#   Prowadzący: mgr inż. Mikołaj Markiewicz
#   Wykonał: Jan Kaniuka
#   Numer indeksu: 303762

import sys, os
from csv import reader
from math import sqrt, pi, exp, floor
from random import seed, shuffle
from numpy import array_split

# basic statistical measures
mean = lambda array: sum(array)/len(array)
std_dev = lambda array: sqrt(sum([(value-mean(array))**2 for value in array]) / (len(array)-1))
gaussian_PDF = lambda x, mean, stddev: (1 / (sqrt(2 * pi) * stddev)) * exp(-((x-mean)**2 / (2 * stddev**2 )))

# Read data in .csv format (with/without header)
def read_data(path, hasHeader, delimiter_type):
  lines = reader(open(path, "r"), delimiter = delimiter_type)
  dataset = list(lines)
  if hasHeader:
    dataset.pop(0)
  for record in dataset:
    for i in range(len(record)):
      record[i] = float(record[i].strip())
  return dataset

# Divide data into k sets for cross validation
def k_cross_validation(dataset, sets_number):
  if sets_number == 1:
    raise TypeError("Minimum number of sets for k-cross validation is 2!")
  local_copy = list(dataset)
  shuffle(local_copy)
  result = array_split(local_copy, sets_number)
  return 	[element.tolist() for element in [*result]]

# Divide set into training and test set based on given division ratio eg. 60/40
def train_and_test_set(dataset, division_ratio=0.60):
  local_copy = list(dataset)
  shuffle(local_copy)
  train_size = floor(division_ratio * len(dataset))
  return [local_copy[:train_size], local_copy[train_size:]]

# Stats for atributes for each class
def stats_for_classes(dataset):
  classes, statistics = dict(), dict()
  for i in range(len(dataset)):
    record = dataset[i]
    wine_quality = record[len(record)-1]
    if (wine_quality not in classes):
      classes[wine_quality] = list()
    classes[wine_quality].append(record)
  for wine_quality, records in classes.items():
    result, stats = [], []
    for j in range(len(records[0])):
      for i in range(len(records)):
        stats.append(records[i][j])
      result.append((mean(stats), std_dev(stats), len(records)))
      stats = []
    result.pop(-1)
    statistics[wine_quality] = result
  return statistics

# Naive Bayes classifier
def NBClassifier(training_set, test_set):
  statistics = stats_for_classes(training_set)
  predictions = list()
  for element in test_set:
    predictions.append(choose_fittest_class(statistics, element))
  return predictions

# Probabilities for each class -> choose argmax(propabilities)
def class_probabilities(class_stats, record):
  count = sum([class_stats[_class][0][2] for _class in class_stats])
  probabilities_list = dict()
  for wine_quality, atribute_stats in class_stats.items():
    probabilities_list[wine_quality] = class_stats[wine_quality][0][2]/count
    for i in range(len(atribute_stats)):
      mean, stdev, num  = atribute_stats[i]
      probabilities_list[wine_quality] *= gaussian_PDF(record[i], mean, stdev)
  return probabilities_list

# Get best prediction about class (wine quality)
def choose_fittest_class(summaries, row):
  probabilities = class_probabilities(summaries, row)
  quality, max_prob = None, -1
  for _class, prob in probabilities.items():
    if quality is None or prob > max_prob:
      max_prob = prob
      quality = _class
  return quality

# Evaluate an algorithm 
def validation(dataset, resampling_type,  k = 5):
  correct_match = 0
  if resampling_type == 'cross_validation':
    sets = k_cross_validation(dataset, k)
  if resampling_type == 'test_and_train':
    sets = train_and_test_set(dataset, 0.6)
  accuracy_list = []
  for set in sets:
    correct_match = 0
    train_set = list(sets)
    train_set.remove(set)
    train_set = sum(train_set, [])
    test_set = list()
    for line in set:
      test_set.append(list(line))
      list(line)[-1] = None
    assumed = NBClassifier(train_set, test_set)
    real = [line[-1] for line in set]
    for i in range(len(real)):
      if real[i] == assumed[i]: correct_match += 1
    accuracy_list.append(correct_match / len(real) * 100)
  return accuracy_list


if __name__ == "__main__":
  seed(1)
  path = os.path.join(sys.path[0], 'winequality-red.csv')
  dataset = read_data(path, hasHeader = True, delimiter_type = ";")
  division_type = 'test_and_train' #  'cross_validation' or 'test_and_train'
  accuracy_list = validation(dataset, division_type )
  print('Resampling method: ', division_type)
  print('Accuracy for each set: %s' % accuracy_list)
  print('Mean accuracy value: %f' %  (sum(accuracy_list)/(len(accuracy_list))))
