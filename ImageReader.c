#include "ImageReader.h"

void read_mnist(double **arr, int amount)
{
    FILE *file = fopen("./data/t10k-images.idx3-ubyte", "rb");
    if (file)
    {
        int magic_number = 0;
        int amount_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        fread((char*) &magic_number, sizeof(magic_number), 1, file); 
        magic_number = change_endian(magic_number);

        fread((char*) &amount_images, sizeof(amount_images), 1, file);
        amount_images = change_endian(amount_images);

        fread((char*) &n_rows, sizeof(n_rows), 1, file);
        n_rows = change_endian(n_rows);

        fread((char*) &n_cols, sizeof(n_cols), 1, file);
        n_cols = change_endian(n_cols);

        for (int i = 0; i < min(amount_images, amount); ++i)
        {
            for (int r = 0; r < n_rows; ++r)
            {
                for (int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    fread((char*) &temp, sizeof(temp), 1, file);
                    arr[i][n_cols * r + c] = (double) temp;
                }
            }
        }
    }

    fclose(file);
}

void print_image(double image[])
{
    char *scale = " .:-=+*#%@";
    for (int i = 0; i < 28; ++i)
    {
        for (int j = 0; j < 28; ++j)
        {
            printf("%c", scale[(int)(image[28 * i + j] * 10)]);
        }
        printf("\n");
    }
}

int main()
{
    double arr[10000][28 * 28];
    read_mnist(arr, 10000);

    print_image(arr[0]);

    return 0;
}


// oi rafa
// precisa só terminar de debugar e ver se esse print image funciona
// corrigir os warnings e tals
// mas acho que ta certo =)