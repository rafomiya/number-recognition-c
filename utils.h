#include <stdio.h>
#include <stdlib.h>
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define square(a) ((a) * (a))
#define swap(a, b, type)             \
    do                               \
    {                                \
        a = (type)((int)a ^ (int)b); \
        b = (type)((int)b ^ (int)a); \
        a = (type)((int)a ^ (int)b); \
    } while (0)

int change_endian(int x);
int arr_max(int *arr, int p, int r);
double random();
void shuffle(double **arr1, char *arr2, int n);
