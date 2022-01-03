import math
import random
from csv import reader
import sys, os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import copy

class Data:
    def __init__(self, filename, has_header, delimiter_type):
        self.filename = filename
        self.path  = os.path.join(sys.path[0], self.filename)
        self.delimiter_type = delimiter_type
        self.has_header = has_header
        self.dataset = None

    # Reads CSV file, converts attributes to floats and class values to integers
    def read_data(self):
        lines = reader(open(self.path, "r"), delimiter = self.delimiter_type)
        self.dataset = list(lines)
        if self.has_header:
            self.dataset.pop(0)
        for record in self.dataset:
            for i in range(len(record)-1):
                record[i] = float(record[i].strip())
            record[-1] = int(record[-1].strip())

    # Remaps class values - assigns successive integers
    # Example of function return: {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5}
    # When expected output is [ 0 0 0 0 1 0] it means class = 7
    def convert_class_values(self):
        all_classes = [record[-1] for record in self.dataset]
        unique_classes = set(all_classes)
        conversion_table = dict()
        for i, class_val in enumerate(unique_classes):
          conversion_table[class_val] = i
        for row in self.dataset:
          row[-1] = conversion_table[row[-1]]
        return conversion_table

    # Performs data normalization -> y = (x - min) / (max - min)
		# output is between 0 and 1
    def normalize_dataset(self):
        minmax = []
        for column in zip(*self.dataset):
            minmax.append([min(column), max(column)])
        for row in self.dataset:
            for i in range(len(row)-1):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    # Prepares data for training
    def prepare_data(self):
        self.read_data()
        self.normalize_dataset()
        self.convert_class_values()

    
class Neuron:
    def __init__(self, weights_number):
        # Init wights with rand numbers between 0 and 1, last weight is bias
        self.weights = [random.uniform(0, 1) for _ in range(weights_number)]
        self.output = 0.0
        self.error = None

    def activate(self, inputs):
        result = 0
        result += self.weights[-1]
        for i, input in enumerate(inputs):
            result += input * self.weights[i]
        return self.activation_function(result)

		# sigmoidal activation function
    def activation_function(self, sum_result):
        return 1.0 / (1.0 + math.exp(-sum_result))

    def activation_function_derivative(self):
        return self.output * (1 - self.output)


class Layer:
    def __init__(self, name, neurons_number, inputs_number):
        self.name = name
        # List of neurons in layer. +1 weight for bias
        self.neurons = [Neuron(inputs_number + 1) for _ in range(neurons_number)]


class NeuralNetwork:
    def __init__(self, input_number, hidden_number, output_number):
        self.layers = [Layer("hidden", hidden_number, input_number), Layer("output", output_number, hidden_number)]
        self.input_number = input_number
        self.hidden_number = hidden_number
        self.output_number = output_number
        self.learn_history = []

    def forward_propagation(self, inputs):
        for layer in self.layers:
            next_layer_inputs = []
            for neuron in layer.neurons:
                neuron.output = neuron.activate(inputs)
                next_layer_inputs.append(neuron.output)
            inputs = next_layer_inputs
        return inputs

    def backward_propagation(self, expected_outputs):
        for i, layer in reversed(list(enumerate(self.layers))):
            errors = []
            if i != 1:  # if layer is not output layer
                for j in range(len(layer.neurons)):
                    error = 0.0
                    for neuron in self.layers[i + 1].neurons:
                        error += neuron.weights[j] * neuron.error
                    errors.append(error)
            else:  # layer is output layer (last layer, first iteration in forloop)
                for j, neuron in enumerate(layer.neurons):
                    errors.append(neuron.output - expected_outputs[j])
            for j, neuron in enumerate(layer.neurons):
                neuron.error = errors[j] * neuron.activation_function_derivative()

    def update_weights(self, data, learn_rate):
        for i, layer in enumerate(self.layers):
            inputs = data[:-1]
            if i != 0:  # layer is not first hidden
                inputs = [neuron.output for neuron in self.layers[i - 1].neurons]
            for neuron in self.layers[i].neurons:
                for j in range(len(inputs)):
                    neuron.weights[j] -= learn_rate * neuron.error * inputs[j]
                neuron.weights[-1] -= learn_rate * neuron.error


    def train(self, epochs, learn_rate, train_data, plot):
        errors_list = []
        for epoch in range(epochs):
            error_sum = 0.0
            for data in train_data:
                outputs = self.forward_propagation(data)
                expected_outputs = [0 for _ in range(self.output_number)]
                expected_outputs[data[-1]] = 1
                error_sum += sum([(expected_outputs[i] - outputs[i]) ** 2 for i in range(len(expected_outputs))])
                self.backward_propagation(expected_outputs)
                self.update_weights(data, learn_rate)
            errors_list.append(error_sum)
            self.learn_history.append(error_sum)

    def predict(self, data):
        outputs = self.forward_propagation(data)
        return outputs.index(max(outputs))


class Test:
    def __init__(self, csv_data, network):
      self.csv_data = csv_data
      self.network = network

    # Divides data into k sets for k-cross validation
    # Return: list of lists()
    def k_cross_validation_split(self, sets_number):
      if sets_number == 1:
        raise Exception("Minimum number of sets for k-cross validation is 2!")
      local_copy = list(self.csv_data.dataset)
      random.shuffle(local_copy)
      result = np.array_split(local_copy, sets_number)
      splitted = [element.tolist() for element in [*result]]
      for sublist in splitted:
        for i in range(len(sublist)):
          sublist[i][-1] = int(sublist[i][-1]) #class value must be int (numpy forces floats)
      return splitted

    # Division into  training and testing set according to a given proportion
    def train_and_test_set_split(self, division_ratio=0.60):
      local_copy = list(self.csv_data.dataset)
      random.shuffle(local_copy)
      train_size = math.floor(division_ratio * len(self.csv_data.dataset))
      return [local_copy[:train_size], local_copy[train_size:]]

    # Calculates accuracy as a percentage (compares 2 lists)
    def calculate_performance(self, actual, predicted):
      correct = 0
      for i in range(len(actual)):
        if actual[i] == predicted[i]:
          correct += 1
      return correct / float(len(actual)) * 100.0


    # Executing simple testing procedure
    def test_network(self, resampling_type, epochs, learning_rate, k_sets = 5, division_ratio = 0.7):
      self.networks = [copy.deepcopy(self.network) for _ in range(k_sets)]
      if resampling_type == "k_cross":
        data = self.k_cross_validation_split(k_sets)
      elif resampling_type == "test&train":
        data = self.train_and_test_set_split(division_ratio)
      accuracy_list = []
      for i, set in enumerate(data):
        train_set = list(data)
        train_set.remove(set)
        train_set = sum(train_set, [])
        test_set = list()
        for line in set:
          test_set.append(list(line))
          list(line)[-1] = None
        self.networks[i].train(epochs, learning_rate, train_set, plot = False)
        predicted, actual = [], []
        actual = [line[-1] for line in set]
        for row in test_set:
          predicted.append(self.networks[i].predict(row))
        accuracy_list.append(self.calculate_performance(actual, predicted))
      return accuracy_list


random.seed(1)
csv_dataset = Data(filename = 'winequality-red.csv', has_header = True, delimiter_type= ";")
csv_dataset.prepare_data()

# 11 neurons in the input layer (wine is described by 11 attributes)
# 19 neurons in the hidden layer as rule of thumb (2/3 * inputs + outputs)
# 11 neurons in the output layer (classes from 0 to 10)

legends = []
for i, learn_rate in enumerate([0.01, 0.1, 0.2, 0.5, 1, 10]):

    network = NeuralNetwork(11,19,11)
    epochs = 50
    tester = Test(csv_dataset, network)
    acc_list = tester.test_network("k_cross", epochs = epochs, learning_rate = learn_rate, k_sets=3)
    print(acc_list)
    acc = sum(acc_list)/len(acc_list)
    plt.plot(tester.networks[0].learn_history)
    legends.append(f"learn_rate={learn_rate}, accuracy={round(acc, 2)}%")

plt.xlabel("epoch")
plt.ylabel("error")
plt.title("Błąd uczonej sieci w kolejnych epokach dla różnych współczynników uczenia")
plt.legend(legends, loc='upper right')
plt.show()
