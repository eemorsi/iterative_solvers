#include <cstdlib>
#include <stdio.h>
#include <algorithm>


#define SWAP(a, b) { typeof(a) temp = a; a = b; b = temp; }
void sort(double *p, int *a, int i, int j, int size)
{
    if (i < size)
    {
        if (j < size - i - 1)
        {
            if (a[j] > a[j + 1])
            {
                // int temp = p[j];
                // p[j] = p[j + 1];
                // p[j + 1] = temp;
                SWAP(p[i], p[j]);
            }
        }
        else
        {
            j = 0;
            ++i;
        }

        sort(p, a, i, ++j, size);
    }
}

int main(int argc, char const *argv[])
{
    int size = 10;

    int rows[] =      {3,   4,    2,   2,      1,    1,   0,    0,  2,   4};
    double values[] = {1.2, 22.3, 6.4, 1.0052, 0.22, 6.1, 7.02, 52, 1.2, 10};

    sort(values, &values[0]+size, [&](int i,int j){return rows[i]<rows[j];});
    sort(values, rows, 0,0, size);
    for (size_t i = 0; i < size; i++)
    {
        printf("%f, ", values[i]);
    }
    

    return 0;
}
