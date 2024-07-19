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

/**
 * Receives an integer and returns it with the endianess swapped.
 */
int change_endian(int x);

/**
 * Seeks the maximum value of the subarray of `arr` going from `p` (inclusive)
 * to `r` (exclusive) and returns it.
 */
int arr_max(int *arr, int p, int r);

/**
 * Returns a random number on the interval [0, 1).
 */
double get_random();
