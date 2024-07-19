#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include "Dataset.h"
#define FOR_W_B(code_w, code_b)                                      \
    do                                                               \
    {                                                                \
        for (int l_ = 1; l_ < nn->amount_layers; ++l_)               \
        {                                                            \
            for (int i_ = 0; i_ < nn->layer_sizes[l_]; ++i_)         \
            {                                                        \
                for (int j_ = 0; j_ < nn->layer_sizes[l_ - 1]; ++j_) \
                {                                                    \
                    code_w;                                          \
                }                                                    \
                code_b;                                              \
            }                                                        \
        }                                                            \
    } while (0)

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

    // refers to ∂C/(∂a^(l+1))
    double *previous_grad_a;
} NeuralNetwork;

/**
 * Initializes a NeuralNetwork with the given `amount_layers` and
 * `layer_sizes`, and returns a pointer to it.
 */
NeuralNetwork *create_neural_network(int amount_layers, int *layer_sizes);

/**
 * Sets the output layer of the given NeuralNetwork as the output to a given
 * input, considering the current weights and biases. In other words, "runs"
 * the NN with the given input.
 */
void compute(NeuralNetwork *nn, double *input);

/**
 * Trains the NeuralNetwork using the stochastic gradient descent methodology
 * to minimize the cost function associated with the output layer.
 * 
 * @param nn a pointer to the NeuralNetwork
 * @param dataset a pointer to the Dataset used for trainment
 * @param learning_rate the rate to scale each iterative step (usually
 * defaults to .01)
 * @param batch_size the amount of images used to calculate each step
 */
void stochastic_gradient_descent(NeuralNetwork *nn, Dataset *dataset, double learning_rate, int batch_size);

/**
 * Frees all the dinamically allocated memory associated with a NeuralNetwork
 */
void destruct_neural_network(NeuralNetwork *nn);
