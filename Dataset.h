#include "utils.h"
#define READ_BINARY "rb"
#define PATH_TRAINING_IMAGES "./data/train-images.idx3-ubyte"
#define PATH_TRAINING_LABELS "./data/train-labels.idx1-ubyte"
#define PATH_TESTING_IMAGES "./data/t10k-images.idx3-ubyte"
#define PATH_TESTING_LABELS "./data/t10k-labels.idx1-ubyte"

typedef enum
{
    TESTING,
    TRAINING
} DatasetType;

typedef struct
{
    FILE *images_file;
    FILE *labels_file;
    double **images;
    char *labels;
    int n;
} Dataset;

Dataset *get_dataset(DatasetType type);
void load_dataset(Dataset *dataset);
void destruct_dataset(Dataset *dataset);
void print_image(double *image);
void shuffle(Dataset *dataset);
