#include "utils.h"

int change_endian(int x)
{
    unsigned char c1, c2, c3, c4;

    c1 = x & 255;
    c2 = (x >> 8) & 255;
    c3 = (x >> 16) & 255;
    c4 = (x >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

int arr_max(int *arr, int p, int r)
{
    int result = arr[p];
    for (int i = p + 1; i < r; ++i)
    {
        if (result < arr[i])
        {
            result = arr[i];
        }
    }

    return result;
}

double random()
{
    return ((double)rand()) / ((double)RAND_MAX);
}

void shuffle(double **arr1, char *arr2, int n)
{
    double *temp1;
    char temp2;

    int j;
    for (int i = n - 1; i > 0; --i)
    {
        j = rand() % (i + 1);

        temp1 = arr1[j];
        arr1[j] = arr1[i];
        arr1[i] = temp1;

        temp2 = arr2[j];
        arr2[j] = arr2[i];
        arr2[i] = temp2;
    }
}
