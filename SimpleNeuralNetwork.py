import numpy as np


class NeuralNetwork:
    def __init__(self, layerSizes):
        """
        Initializes a neural network with given layer sizes and random weights.

        Args:
        - layerSizes (list of int): Specifies the number of neurons in each layer of the network.

        Returns:
        None
        """

        # Store the layer sizes as an instance variable.
        self.layerSizes = layerSizes

        # Initialize the synaptic weights between each layer randomly.
        self.synapticWeights = [2 * np.random.random((self.layerSizes[i], self.layerSizes[i + 1])) - 1
                                for i in range(len(self.layerSizes) - 1)]

    @staticmethod
    def sigmoid(x):
        """
        Calculates the sigmoid function for a given input.

        Args:
        x (list of float): Input to the sigmoid function.

        Returns:
        list of float: Output of the sigmoid function, which is a value between 0 and 1.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoidDerivative(x):
        """
        Calculates the derivative of the sigmoid function for a given input.

        Args:
        x (list of float): Input to the sigmoid function.

        Returns:
        list of float: Output of the derivative of the sigmoid function.
        """
        return x * (1 - x)

    def train(self, inputs, outputs, numIterations):
        """
        Trains the neural network using the input data and desired output data.

        Args:
        inputs (list of float): The training input data.
        outputs (list of float): The desired output data.
        numIterations (int): The number of times to run the training process.

        Returns:
        None
        """
        for i in range(numIterations):
            # Feed forward
            inputPrediction = [inputs]  # initialize with input data
            for layerWeights in self.synapticWeights:
                # calculate output for each layer by passing previous layer output through the weights
                inputPrediction.append(self.sigmoid(np.dot(inputPrediction[-1], layerWeights)))

            # Backpropagation
            error = outputs - inputPrediction[-1]  # Calculate the error between the output and the predicted output
            deltas = [error * self.sigmoidDerivative(inputPrediction[-1])]  # Calculate the delta for the output layer
            for layer in range(len(inputPrediction) - 2, 0, -1):  # Iterate over the hidden layers in reverse order
                deltas.append(np.dot(deltas[-1], self.synapticWeights[layer].T) * self.sigmoidDerivative(inputPrediction[layer]))
            deltas.reverse()  # Reverse the order of the deltas to match the order of the layers in the network

            # Weight update
            for j in range(len(self.synapticWeights)):
                # Update the weights using the deltas and the activations
                self.synapticWeights[j] += np.dot(inputPrediction[j].T, deltas[j])

    def predict(self, inputs):
        """
        Predict the output value of the user input accordingly to what the ANN was train for

        Args:
        input (list of float): User inputs

        Returns:
        output (list of float): ANN predict output values
        """
        outputs = inputs  # Initialize with user inputs
        for layerWeights in self.synapticWeights:
            outputs = self.sigmoid(np.dot(outputs, layerWeights))
        return outputs


if __name__ == "__main__":

    # Training set
    trainingSetInputs = np.array([[0.78], [0.80], [0.33], [0.93], [0.91], [0.74], [0.34], [0.59], [0.068], [0.12],
                                  [0.42], [0.48], [0.71], [0.38], [0.072], [0.47], [0.24], [0.19], [0.078], [0.55]])
    trainingSetOutputs = np.array([[0.078, 0.080, 0.033, 0.093, 0.091, 0.074, 0.034, 0.059, 0.0068, 0.012,
                                    0.042, 0.048, 0.071, 0.038, 0.0072, 0.047, 0.024, 0.019, 0.0078, 0.055]]).T

    nn = NeuralNetwork([1, 5, 5, 1])

    print("\nSynaptic weight before training:")
    print(nn.synapticWeights)

    numIterations = 10000
    nn.train(trainingSetInputs, trainingSetOutputs, numIterations)

    print("\nSynaptic weight after training:")
    print(nn.synapticWeights)

    userValue = input("\nEnter a value between 1 and 0: ")
    print("predicted value:", nn.predict(np.array([float(userValue)]))[0])
