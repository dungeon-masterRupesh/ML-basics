import random

import numpy as np

class Network(object):

    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
    """
    From one layer of neuron to another layer
    Starting from initial input given to the neuron
    We will keep it updating layerWise to get final output
    """
    def feedforward(self,a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a

    """
    Stochiastic gradient descent algo applied
    finding a random set of data to find grad in cost function
    we call update of the network with given mini_data_set
    """
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Iteration {0}: {1} / {2}".format(j+1, self.evaluate(test_data), n_test) #test_data is never used for training
            else:
                print "Iteration {0} complete".format(j+1)

    def update_mini_batch(self,mini_batch,eta):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            """
            feedforward with x, backpropagation with y
            Non-vectorised implementation
            Doing for each data sample step by step
            """
            step_b, step_w = self.backprop(x, y)
            delta_w = [d+s for d, s in zip(delta_w, step_w)]
            delta_b = [d+s for d, s in zip(delta_b, step_b)]
  
        # updating weights of network
        # step for each batch updated with multipling something that decides step_size
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, delta_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, delta_b)]

    def backprop(self, x, y):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        """
        actually the implementation is same as on done for neural net in coursera example. Here cost-fuction is MSE / 2
        so cost grad of x-y^2 / 2 is x-y
        formula 
        d3 = a3 - y

        """
        delta = self.cost_derivative(activations[-1], y) * sigmoid_grad(zs[-1])
        delta_w[-1] = np.dot(delta, activations[-2].transpose())
        delta_b[-1] = delta
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_grad(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            delta_w[-l] = np.dot(delta, activations[-l-1].transpose())
            delta_b[-l] = delta
        return (delta_b, delta_w)

    def evaluate(self,test_data):
        result = [
            (np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
        return sum(int(x == y) for (x,y) in result)


    def cost_derivative(self, output_activations,y):
        # in our case of cost function it's just a - x i.e.
        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_grad(z):
    return sigmoid(z)*(1-sigmoid(z))
