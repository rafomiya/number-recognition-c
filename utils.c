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

double get_random()
{
    return rand() / (RAND_MAX + 1.0);
}
