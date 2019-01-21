from numpy import exp, array, random, dot
from pip._vendor.urllib3.connectionpool import xrange
import pandas as pd  # A beautiful library to help us work with data as tables


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
    
        self.synaptic_weights = random.random((12, 2))
        self.synaptic_weights = random.random((12, 2))

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron)
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":
    # Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    dataframe = pd.read_csv("datasetcsv.csv")
    # remove columns we don't care about
    dataframe = dataframe.drop(["unknown.17", "unknown.16", "unknown.15", "unknown.14", "unknown.13", "unknown.12",
                                "unknown.11", "unknown.10", "unknown.9", "unknown.8", "unknown.7", "unknown.6",
                                "unknown.5"
                                   , "unknown.4", "unknown.3", "unknown.2", "unknown.1", "unknown"],
                               axis=1)
    # We'll only use the first 10 rows of the dataset in this example
    # dataframe = dataframe[0:30]
    # Let's have the notebook show us how the dataframe looks now

    # print(dataframe)

    inputX = dataframe.loc[:,
             ['radius ', 'texture', 'perimeter', 'area', 'smoothness ', 'compactness ', 'concavity', 'concave  '
                 , 'points', 'symmetry', 'fractal', ' dimension']].values

    inputY = dataframe.loc[:, ["Label"]].values
    for x in range(inputY.size):
        if (inputY[x] == "M"):
            inputY[x] = 1
        else:
            inputY[x] = 0

    # print(dataframe)

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = inputX
    training_set_outputs = inputY
    # .T

    # print(training_set_outputs)

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    # Test the neural network with a new situation.
    print("Considering new situation [1, 0, 0] -> ?: ")
    print(neural_network.think(array([1, 0, 0])))
