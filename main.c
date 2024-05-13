#include "NeuralNetwork.h"

void get_result(NeuralNetwork *nn, Dataset dataset, int sample);

int main()
{
    int amount_layers = 4;
    int layer_sizes[4] = {28 * 28, 16, 16, 10};
    Dataset training = get_dataset(TRAINING);
    Dataset testing = get_dataset(TESTING);

    NeuralNetwork *nn = create_neural_network(amount_layers, layer_sizes);

    load_dataset(&training);
    load_dataset(&testing);

    // for (int i = 0; i < 10; ++i)
    // {
    //     printf("[%d]\n", testing.labels[i]);
    //     print_image(testing.images[i]);
    // }

    stochastic_gradient_descent(nn, training, 0.01, 100);

    for (int i = 0; i < 10; ++i)
    {
        get_result(nn, testing, i);
    }

    destruct_dataset(training);
    destruct_dataset(testing);
    destruct_neural_network(nn);

    return 0;
}

void get_result(NeuralNetwork *nn, Dataset dataset, int sample)
{
    compute(nn, dataset.images[sample]);
    printf("[%d]\n", dataset.labels[sample]);
    for (int i = 0; i < 10; ++i)
    {
        printf("%d,\t", i);
    }
    printf("\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("%lf,\t", nn->activations[3][i]);
    }
    printf("\n\n");
}
