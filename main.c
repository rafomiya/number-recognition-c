#include "NeuralNetwork.h"

int main()
{
    int amount_layers = 4;
    int *layer_sizes = {29 * 29, 16, 16, 10};

    NeuralNetwork *nn = create_neural_network(amount_layers, layer_sizes);
}
