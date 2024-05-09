#include <stdlib.h>
#include <math.h>
#include "NeuralNetwork.h"

int max(int *arr, int n)
{
    int result = arr[0];
    for (int i = 1; i < n; ++i)
    {
        if (result < arr[i])
        {
            result = arr[i];
        }
    }

    return result;
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double get_activation(NeuralNetwork *nn, Node *node)
{
    if (nn->activations[node->layer][node->index] != -1)
    {
        return nn->activations[node->layer][node->index];
    }

    if (node->layer == 0)
    {
        // get from input data or generate random
        return ((double)rand()) / ((double)RAND_MAX);
    }

    double z = 0;
    for (int i = 0; i < nn->layer_sizes[node->layer - 1]; ++i)
    {
        z += nn->weights[node->layer - 1][i][node->index] * nn->activations[node->layer - 1][i];
    }
    z /= nn->layer_sizes[node->layer - 1];

    z -= nn->biases[node->layer][node->index];

    return sigmoid(z);
}

NeuralNetwork *create_neural_network(int amount_layers, int *layer_sizes)
{
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    nn->amount_layers = amount_layers;
    nn->layer_sizes = layer_sizes;
    nn->max_layer_size = max(layer_sizes, amount_layers);

    nn->activations = malloc(amount_layers * sizeof(double *));
    nn->weights = malloc(amount_layers * sizeof(double **));
    nn->biases = malloc(amount_layers * sizeof(double *));

    for (int i = 0; i < amount_layers; ++i)
    {
        nn->activations[i] = malloc(layer_sizes[i] * sizeof(double));
        for (int j = 0; j < layer_sizes[i]; ++j)
        {
            nn->activations[i][j] = -1;
        }

        nn->weights[i] = malloc(layer_sizes[i] * sizeof(double *));
        for (int j = 0; j < layer_sizes[i]; ++j)
        {
            nn->weights[i][j] = malloc(layer_sizes[i + 1] * sizeof(double));
            for (int k = 0; k < layer_sizes[j + 1]; ++k)
            {
                nn->weights[i][j][k] = -1;
            }
        }

        nn->biases[i] = malloc(layer_sizes[i] * sizeof(double));
        for (int j = 0; j < layer_sizes[i]; ++j)
        {
            nn->biases[i][j] = -1;
        }
    }

    return nn;
}

// implement cost function to receive an int (0-9) representing the right result
// and make the sum of squared differences of with the activations of the last layer

// prepare to import mnist database of handwritten digits and make the image input to
// memory vectors
