#include <stdlib.h>
#include <math.h>
#include "utils.h"

typedef struct
{
    int layer;
    int index;
} Node;

typedef struct
{
    int amount_layers;
    int *layer_sizes;

    double **activations;
    double ***weights;
    double **biases;

    // activations[amount_layers][max_layer_size];
    // weights[amount_layers][max_layer_size][max_layer_size];
    // biases[amount_layers][max_layer_size];

    int max_layer_size;
} NeuralNetwork;

double get_activation(NeuralNetwork *nn, Node *node);
NeuralNetwork *create_neural_network(int amount_layers, int *layer_sizes);
