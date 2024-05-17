#include "NeuralNetwork.h"
#define AMOUNT_LAYERS 4

void get_result(NeuralNetwork *nn, Dataset *dataset, int sample);

int max_index(double *arr, int n)
{
    int index = 0;
    for (int i = 1; i < n; ++i)
    {
        if (arr[i] > arr[index])
        {
            index = i;
        }
    }

    return index;
}

int test(NeuralNetwork *nn, Dataset *testing);

int main()
{
    int layer_sizes[AMOUNT_LAYERS] = {28 * 28, 36, 32, 10};
    Dataset *training = get_dataset(TRAINING);
    Dataset *testing = get_dataset(TESTING);

    NeuralNetwork *nn = create_neural_network(AMOUNT_LAYERS, layer_sizes);

    load_dataset(training);
    load_dataset(testing);

    for (int i = 0; i < 8; ++i)
    {
        stochastic_gradient_descent(nn, training, 0.01, 100);
    }

    test(nn, testing);

    destruct_dataset(training);
    destruct_dataset(testing);
    destruct_neural_network(nn);

    return 0;
}

void get_result(NeuralNetwork *nn, Dataset *dataset, int sample)
{
    int d;

    compute(nn, dataset->images[sample]);

    printf("[%d]\n", dataset->labels[sample]);

    for (int i = 0; i < 10; ++i)
    {
        printf("%d\t", i);
        d = (int)(50 * nn->activations[AMOUNT_LAYERS - 1][i]);
        for (int j = 0; j < d; ++j)
        {
            printf("$");
        }
        printf("\n");
    }

    printf("\n");
}

// returns the amount of correct samples
int test(NeuralNetwork *nn, Dataset *testing)
{
    int result, last_layer_size = nn->layer_sizes[AMOUNT_LAYERS - 1];
    int total = testing->n;
    int amount_correct = 0;

    for (int i = 0; i < testing->n; ++i)
    {
        compute(nn, testing->images[i]);

        result = max_index(nn->activations[AMOUNT_LAYERS - 1], last_layer_size);

        if (testing->labels[i] == result)
        {
            amount_correct += 1;
        }
        else if (random() < 0.1) // 10% chance
        {
            print_image(testing->images[i]);
            get_result(nn, testing, i);
            printf("\n\n\n");
        }

        printf("Accuracy: %f%\n", 100.0 * (double)amount_correct / (double)(i + 1));
    }

    return amount_correct;
}
