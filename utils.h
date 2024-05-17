#include <stdio.h>
#include <stdlib.h>
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define square(a) ((a) * (a))
#define swap(a, b, type) \
    do                   \
    {                    \
        type temp_ = a;  \
        a = b;           \
        b = temp_;       \
    } while (0)

int change_endian(int x);
int arr_max(int *arr, int p, int r);
double random();
