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

    // refers to ∂C/(∂a^(l))
    double *current_grad_a;
    double *previous_grad_a;
} NeuralNetwork;

NeuralNetwork *create_neural_network(int amount_layers, int *layer_sizes);
void compute(NeuralNetwork *nn, double *input);
void stochastic_gradient_descent(NeuralNetwork *nn, Dataset dataset, double learning_rate, int mini_batch_size);
void destruct_neural_network(NeuralNetwork *nn);
