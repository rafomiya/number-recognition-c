#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include "ImageReader.h"

typedef struct
{
    int amount_layers;
    int *layer_sizes;

    // activations[l][i]: activation of the i-th neuron of the l-th layer
    double **activations;

    // weights[l][i][j]: weight connecting the j-th neuron of the (l-1)-th
    // layer and the i-th neuron of the l-th layer
    double ***weights;

    // biases[l][i]: bias of the i-th neuron of the l-th layer
    double **biases;
} NeuralNetwork;

void set_activation(NeuralNetwork *nn, int layer, int index);
NeuralNetwork *create_neural_network(int amount_layers, int *layer_sizes);
// void stochastic_gradient_descent(NeuralNetwork *nn, double training_data[][], double labels[], int n, double learning_rate, int mini_batch_size);
