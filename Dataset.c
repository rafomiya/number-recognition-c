#include "Dataset.h"

Dataset *get_dataset(DatasetType type)
{
    Dataset *dataset = malloc(sizeof(Dataset));
    int trash;
    int amount;

    if (type == TRAINING)
    {
        dataset->images_file = fopen(PATH_TRAINING_IMAGES, READ_BINARY);
        dataset->labels_file = fopen(PATH_TRAINING_LABELS, READ_BINARY);
    }
    else
    {
        dataset->images_file = fopen(PATH_TESTING_IMAGES, READ_BINARY);
        dataset->labels_file = fopen(PATH_TESTING_LABELS, READ_BINARY);
    }

    // skipping the initial unnecessary bytes

    // ignoring magic number from both files
    fread((char *)&trash, sizeof(int), 1, dataset->images_file);
    fread((char *)&trash, sizeof(int), 1, dataset->labels_file);

    // reading amount of images
    fread((char *)&amount, sizeof(int), 1, dataset->images_file);
    fread((char *)&trash, sizeof(int), 1, dataset->labels_file); // it's same thing on both files
    amount = change_endian(amount);

    // ignoring dimension sizes from the images file
    fread((char *)&trash, sizeof(int), 1, dataset->images_file); // 28
    fread((char *)&trash, sizeof(int), 1, dataset->images_file); // 28

    dataset->images = malloc(amount * sizeof(double *));
    for (int i = 0; i < amount; ++i)
    {
        dataset->images[i] = malloc(789 * sizeof(double));
    }

    dataset->labels = malloc(amount * sizeof(char));
    dataset->n = amount;

    return dataset;
}

void load_dataset(Dataset *dataset)
{
    unsigned char temp;
    for (int i = 0; i < dataset->n; ++i)
    {
        for (int row = 0; row < 28; ++row)
        {
            for (int column = 0; column < 28; ++column)
            {
                fread((char *)&temp, sizeof(char), 1, dataset->images_file);
                dataset->images[i][28 * row + column] = ((double)temp) / 256.0;
            }
        }

        fread((char *)&temp, sizeof(char), 1, dataset->labels_file);
        dataset->labels[i] = temp;
    }
}

void print_image(double *image)
{
    char *scale = " .:-=+*#%@";
    for (int i = 0; i < 28; ++i)
    {
        for (int j = 0; j < 28; ++j)
        {
            printf("%c ", scale[(int)(image[28 * i + j] * 10)]);
        }
        printf("\n");
    }
    printf("\n\n\n");
}

void shuffle(Dataset *dataset)
{
    int j;
    for (int i = dataset->n - 1; i > 0; --i)
    {
        j = rand() % (i + 1);

        swap(dataset->images[j], dataset->images[i], double *);
        swap(dataset->labels[i], dataset->labels[j], char);
    }
}

void destruct_dataset(Dataset *dataset)
{
    fclose(dataset->images_file);
    fclose(dataset->labels_file);
    for (int i = 0; i < dataset->n; ++i)
    {
        free(dataset->images[i]);
    }

    free(dataset->images);
    free(dataset->labels);
    free(dataset);
}
