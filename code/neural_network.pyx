import numpy as np
from parameters import parameters as p
cimport cython
import cmath

cdef class neural_network:
    cdef public int n_rovers = p.data["Number of Agents"]
    cdef public int n_inputs = p.data["Number of Inputs"]
    cdef public int n_outputs = p.data["Number of Outputs"]
    cdef public int n_nodes = p.data["Number of Nodes"]  # Number of nodes in hidden layer
    cdef public int n_weights = (n_inputs + 1)*n_nodes + (n_nodes + 1)*n_outputs
    cdef public double input_bias = 1.0
    cdef public double hidden_bias = 1.0

    weights = np.zeros(n_weights, n_rovers)
    input_layer = np.zeros(n_inputs, n_rovers)
    hidden_layer = np.zeros(n_nodes, n_rovers)
    output_layer = np.zeros(n_outputs, n_rovers)
    cdef double[:] net_weights = weights
    cdef double[:] in_layer = input_layer
    cdef double[:] h_layer = hidden_layer
    cdef double[:] out_layer = output_layer

    cpdef reset_nn(self): # Clear current network
        self.weights = np.zeros(self.n_weights, self.n_rovers)
        self.input_layer = np.zeros(self.n_inputs, self.n_rovers)
        self.hidden_layer = np.zeros(self.n_nodes, self.n_rovers)
        self.output_layer = np.zeros(self.n_outputs, self.n_rovers)
        self.net_weights = self.weights
        self.in_layer = self.input_layer
        self.h_layer = self.hidden_layer
        self.out_layer = self.output_layer

    cpdef get_inputs(self, state_vec, rov_id):  # Get inputs from state-vector
        for i in range(self.n_inputs):
            self.in_layer[rov_id][i] = state_vec[i]

    cpdef get_weights(self, ccea_weights, rov_id):  # Get weights from CCEA population
        for i in range(self.n_weights):
            self.net_weights[rov_id][i] = ccea_weights[i]

    cpdef reset_layers(self, rov_id):  # Clear hidden layers and output layers
        for i in range(self.n_nodes):
            self.h_layer[rov_id][i] = 0.0
        for i in range(self.n_outputs):
            self.out_layer[rov_id][i] = 0.0

    cpdef get_outputs(self, rov_id):
        count = 0
        self.reset_layers(rov_id)

        for i in range(self.n_inputs):  # Pass inputs to hidden layer
            for j in range(self.n_nodes):
                self.h_layer[rov_id][j] += self.in_layer[rov_id][i] * self.weights[rov_id][count]
                count += 1

        for j in range(self.n_nodes):  # Add Biasing Node
            self.h_layer[rov_id][j] += (self.input_bias * self.weights[rov_id][count])
            count += 1

        for i in range(self.n_nodes):  # Pass through sigmoid
            self.h_layer[rov_id][i] = self.sigmoid(self.h_layer[rov_id][i])

        for i in range(self.n_nodes):  # Pass from hidden layer to output layer
            for j in range(self.n_outputs):
                self.out_layer[rov_id][j] += self.h_layer[rov_id][i] * self.weights[rov_id][count]
                count += 1

        for j in range(self.n_outputs):  # Add biasing node
            self.out_layer[rov_id][j] += (self.hidden_bias * self.weights[rov_id][count])
            count += 1

        for i in range(self.n_outputs):  # Pass through sigmoid
            self.out_layer[rov_id][i] = self.sigmoid(self.out_layer[rov_id][i])

        #RETURN OUTPUTS

    cpdef tanh(self, inp): # Tanh function as activation function
        tanh = (2/(1 + cmath.exp(-2*inp)))-1
        return tanh

    cpdef sigmoid(self, inp): # Sigmoid function as activation function
        sig = 1/(1 + cmath.exp(-inp))
        return sig