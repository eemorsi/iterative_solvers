
#include<stdlib.h>
#include<stdio.h>

void alert(double **arr, int size){
    *arr = (double *) aligned_alloc(128, size*sizeof(double));
    for (size_t i = 0; i < size; i++)
    {
        (*arr)[i] = 7;
        // printf("%f\n", arr[i]);
    }
    
}

int main(int argc, char const *argv[])
{
    double* arr; 
    int size =10;
    alert(&arr, size);

    for (size_t i = 0; i < size; i++)
    {
        printf("%f\n", arr[i]);
    }

    return 0;
}
