#include "NeuralNetwork.h"

double y[10][10] = {
    {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};

double relu(double x)
{
    return max(0, x);
}

// given a neuron from the specified layer, returns relu'(z)
double relu_prime(NeuralNetwork *nn, int layer, int index)
{
    return nn->activations[layer][index] > 0;
}

void set_activation(NeuralNetwork *nn, int layer, int index)
{
    if (layer == 0)
    {
        // activations of the first layer are inputs
        return;
    }

    double z = 0;
    for (int i = 0; i < nn->layer_sizes[layer - 1]; ++i)
    {
        z += nn->weights[layer][index][i] * nn->activations[layer - 1][i];
    }

    z += nn->biases[layer][index];

    nn->activations[layer][index] = relu(z);
}

NeuralNetwork *create_neural_network(int amount_layers, int *layer_sizes)
{
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    nn->amount_layers = amount_layers;
    nn->layer_sizes = layer_sizes;

    nn->activations = malloc(amount_layers * sizeof(double *));
    nn->weights = malloc(amount_layers * sizeof(double **));
    nn->biases = malloc(amount_layers * sizeof(double *));

    for (int l = 1; l < amount_layers; ++l)
    {
        nn->weights[l] = malloc(layer_sizes[l] * sizeof(double *));
        nn->biases[l] = malloc(layer_sizes[l] * sizeof(double));
        for (int i = 0; i < layer_sizes[l]; ++i)
        {
            nn->weights[l][i] = malloc(layer_sizes[l - 1] * sizeof(double));
            for (int j = 0; j < layer_sizes[l - 1]; ++j)
            {
                // begins with random value on the interval [-.5, .5)
                nn->weights[l][i][j] = random() - 0.5;
            }

            // begins with 0
            nn->biases[l][i] = 0;
        }

        nn->activations[l] = malloc(layer_sizes[l] * sizeof(double));
    }

    int size_longest_layer = arr_max(layer_sizes, 1, amount_layers);
    nn->current_grad_a = calloc(size_longest_layer, sizeof(double));
    nn->previous_grad_a = calloc(size_longest_layer, sizeof(double));

    return nn;
}

/*
double cost_single_sample(NeuralNetwork *nn, double *data, char result)
{
    compute(nn, data);

    double sum = 0;
    for (int i = 0; i < nn->layer_sizes[nn->amount_layers - 1]; ++i)
    {
        sum += square(nn->activations[nn->amount_layers - 1][i] - y[result][i]);
    }

    return sum;
}

double cost(NeuralNetwork *nn, double **data, char *results, int amount_samples)
{
    double sum = 0;

    for (int i = 0; i < amount_samples; ++i)
    {
        sum += (1 / amount_samples) * cost_single_sample(nn, data[i], results[i]);
    }

    return sum;
}
*/

// apply softmax to last layer
void apply_softmax(NeuralNetwork *nn)
{
    // n: amount of neurons on last layer
    int n = nn->layer_sizes[nn->amount_layers - 1];

    double summation = 0;
    double exps[n];
    for (int i = 0; i < n; ++i)
    {
        exps[i] = exp(nn->activations[nn->amount_layers - 1][i]);
        summation += exps[i];
    }

    for (int i = 0; i < n; ++i)
    {
        nn->activations[nn->amount_layers - 1][i] = exps[i] / summation;
    }
}

// sets the output layer as the current outpupt to a certain input,
// i.e, "runs" the neural network with this given input
void compute(NeuralNetwork *nn, double *input)
{
    nn->activations[0] = input;

    for (int l = 1; l < nn->amount_layers; ++l)
    {
        for (int i = 0; i < nn->layer_sizes[l]; ++i)
        {
            set_activation(nn, l, i);
        }
    }

    apply_softmax(nn);
}

void update_parameters(NeuralNetwork *nn, double ***gradient_w, double **gradient_b, double learning_rate)
{
    FOR_W_B(nn->weights[l_][i_][j_] -= gradient_w[l_][i_][j_] * learning_rate, nn->biases[l_][i_] -= gradient_b[l_][i_] * learning_rate);
}

void free_weights_and_biases(double ***weights, double **biases, int amount_layers, int *layer_sizes)
{
    for (int l = 1; l < amount_layers; ++l)
    {
        for (int i = 0; i < layer_sizes[l]; ++i)
        {
            free(weights[l][i]);
        }

        free(weights[l]);
        free(biases[l]);
    }
    free(weights);
    free(biases);
}

void destruct_neural_network(NeuralNetwork *nn)
{
    for (int l = 1; l < nn->amount_layers; ++l)
    {
        free(nn->activations[l]);
    }
    free(nn->activations);

    free(nn->current_grad_a);
    free(nn->previous_grad_a);

    free_weights_and_biases(nn->weights, nn->biases, nn->amount_layers, nn->layer_sizes);
    free(nn);
}

//
//
// linha divisora entre (certo /\) e (errado \/)
//
//

void backprop_wb(NeuralNetwork *nn, int layer, double *grad_a, double ***gradient_w, double **gradient_b, int mini_batch_size)
{
    double b_increment;
    for (int i = 0; i < nn->layer_sizes[layer]; ++i)
    {
        b_increment = 0;
        if (relu_prime(nn, layer, i))
        {
            b_increment = grad_a[i] / mini_batch_size;
        }

        gradient_b[layer][i] += b_increment;

        for (int j = 0; j < nn->layer_sizes[layer - 1]; ++j)
        {
            gradient_w[layer][i][j] += b_increment * nn->activations[layer - 1][j];
        }
    }
}

void update_grad_a(NeuralNetwork *nn, int l)
{
    if (l <= 1)
    {
        return;
    }

    for (int i = 0; i < nn->layer_sizes[l - 1]; ++i)
    {
        nn->current_grad_a[i] = 0;
        for (int j = 0; j < nn->layer_sizes[l]; ++j)
        {
            if (relu_prime(nn, l, j))
            {
                nn->current_grad_a[i] += nn->previous_grad_a[j] * nn->weights[l - 1][j][i];
            }
        }
    }
}

// this functions makes a += at the gradient matrices
void backpropagation(NeuralNetwork *nn, double ***gradient_w, double **gradient_b, char result, int mini_batch_size)
{
    // loading grad_a_l as a^(L)
    for (int i = 0; i < nn->layer_sizes[nn->amount_layers - 1]; ++i)
    {
        nn->current_grad_a[i] = 2 * (nn->activations[nn->amount_layers - 1][i] - y[result][i]);
    }

    for (int l = nn->amount_layers - 1; l > 0; --l)
    {
        // += ∂C/∂w^l; += ∂C/∂b^l;
        backprop_wb(nn, l, nn->current_grad_a, gradient_w, gradient_b, mini_batch_size);

        // previous <- current
        swap(nn->current_grad_a, nn->previous_grad_a, double *);

        // updates ∂C/∂a^l (current <- new layer)
        update_grad_a(nn, l);
    }
}

// it would be nice if (mini_batch_size | n) is true. please use this function this way.
void stochastic_gradient_descent(NeuralNetwork *nn, Dataset *dataset, double learning_rate, int mini_batch_size)
{
    shuffle(dataset);

    // initializing gradient vectors
    double ***gradient_w = malloc(nn->amount_layers * sizeof(double **));
    double **gradient_b = malloc(nn->amount_layers * sizeof(double *));
    for (int l = 1; l < nn->amount_layers; ++l)
    {
        gradient_w[l] = malloc(nn->layer_sizes[l] * sizeof(double *));
        for (int i = 0; i < nn->layer_sizes[l]; ++i)
        {
            gradient_w[l][i] = malloc(nn->layer_sizes[l - 1] * sizeof(double));
        }

        gradient_b[l] = malloc(nn->layer_sizes[l] * sizeof(double));
    }

    for (int i = 0; i < dataset->n; i += mini_batch_size)
    {
        FOR_W_B(gradient_w[l_][i_][j_] = 0, gradient_b[l_][i_] = 0);

        for (int j = i; j < i + mini_batch_size; ++j)
        {
            compute(nn, dataset->images[j]);

            // essa funçao precisa dar += nos gradientes (scaled multiplicando por 1/minibatchsize)
            backpropagation(nn, gradient_w, gradient_b, dataset->labels[j], mini_batch_size);
        }

        update_parameters(nn, gradient_w, gradient_b, learning_rate);
        printf("Updating weights and biases. Finished training until %dth sample\n", i + mini_batch_size);
    }

    free_weights_and_biases(gradient_w, gradient_b, nn->amount_layers, nn->layer_sizes);
}
