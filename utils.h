#include <stdio.h>
#include <stdlib.h>
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define square(a) ((a) * (a))

int change_endian(int x);
int arr_max(int *arr, int n);
double random();
void shuffle(double **arr1, double **arr2, int n);
