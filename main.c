#include "NeuralNetwork.h"
#define AMOUNT_LAYERS 4
#define EPOCHS 1

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
    int layer_sizes[AMOUNT_LAYERS] = {28 * 28, 16, 16, 10};
    Dataset *training = get_dataset(TRAINING);
    Dataset *testing = get_dataset(TESTING);

    NeuralNetwork *nn = create_neural_network(AMOUNT_LAYERS, layer_sizes);

    load_dataset(training);
    load_dataset(testing);

    for (int i = 0; i < EPOCHS; ++i)
    {
        stochastic_gradient_descent(nn, training, 0.09, 100);
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

void save_configs(NeuralNetwork *nn)
{
    FILE *file = fopen("configs", "w");

    fprintf(file, "%d\n", nn->amount_layers);

    for (int i = 0; i < nn->amount_layers; ++i)
    {
        fprintf(file, "%d ", nn->layer_sizes[i]);
    }
    fprintf(file, "\n");

    for (int l = 1; l < nn->amount_layers; ++l)
    {
        for (int i = 0; i < nn->layer_sizes[l]; ++i)
        {
            for (int j = 0; j < nn->layer_sizes[l - 1]; ++j)
            {
                fprintf(file, "%f ", nn->weights[l][i][j]);
            }
            fprintf(file, "\n");
        }
    }

    for (int l = 1; l < nn->amount_layers; ++l)
    {
        for (int i = 0; i < nn->layer_sizes[l]; ++i)
        {
            fprintf(file, "%f ", nn->biases[l][i]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

NeuralNetwork *load_configs()
{
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));

    FILE *file = fopen("configs", "w");

    fscanf(file, "%d", &nn->amount_layers);

    nn->layer_sizes = malloc(nn->amount_layers * sizeof(int));

    nn->activations = malloc(nn->amount_layers * sizeof(double*));
    nn->weights = malloc(nn->amount_layers * sizeof(double**));
    nn->biases = malloc(nn->amount_layers * sizeof(double*));

    fscanf(file, "%d", &nn->layer_sizes[0]);
    for (int l = 1; l < nn->amount_layers; ++l)
    {
        fscanf(file, "%d", &nn->layer_sizes[l]);
    }

    for (int l = 1; l < nn->amount_layers; ++l)
    {
        nn->activations[l] = malloc(nn->layer_sizes[l] * sizeof(double));
        nn->weights[l] = malloc(nn->layer_sizes[l] * sizeof(double*));
        nn->biases[l] = malloc(nn->layer_sizes[l] * sizeof(double));

        for (int i = 0; i < nn->layer_sizes[l]; ++i)
        {
            nn->weights[l][i] = malloc(nn->layer_sizes[l - 1] * sizeof(double));
            for (int j = 0; j < nn->layer_sizes[l - 1]; ++j)
            {
                fscanf(file, "%f ", &nn->weights[l][i][j]);
            }
        }
    }

    for (int l = 1; l < nn->amount_layers; ++l)
    {
        for (int i = 0; i < nn->layer_sizes[l]; ++i)
        {
            fscanf(file, "%f", &nn->biases[l][i]);
        }
    }

    fclose(file);

    int size_longest_layer = arr_max(nn->layer_sizes, 1, nn->amount_layers);
    nn->current_grad_a = calloc(size_longest_layer, sizeof(double));
    nn->previous_grad_a = calloc(size_longest_layer, sizeof(double));

    return nn;
}

// returns the amount of correct samples
int test(NeuralNetwork *nn, Dataset *testing)
{
    int result, last_layer_size = nn->layer_sizes[AMOUNT_LAYERS - 1];
    int amount_correct = 0;

    FILE *file = fopen("output", "a");

    for (int i = 0; i < testing->n; ++i)
    {
        compute(nn, testing->images[i]);

        result = max_index(nn->activations[AMOUNT_LAYERS - 1], last_layer_size);

        if (testing->labels[i] == result)
        {
            amount_correct += 1;
        }
        else if (get_random() < 0.1) // 10%
        {
            print_image(testing->images[i]);
            get_result(nn, testing, i);
            printf("\n\n\n");
        }

        printf("Accuracy: %f%%\n", 100.0 * (double)amount_correct / (double)(i + 1));
    }

    fprintf(file, "%d/%d = %f%%\n", amount_correct, testing->n, (double)amount_correct / testing->n);
    fclose(file);

    save_configs(nn);

    return amount_correct;
}
