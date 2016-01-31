/*
    Code for 1d convolution
*/
#include<stdio.h>
#include<cuda.h>
#include<errno.h>
#include<assert.h>
#define DEBUG 1
#if not defined(DEBUG)
#define MAX_VAL 800
#define ARR_SIZE (4096*4096*3)
#define MASK_SIZE 11
#else
#define MAX_VAL 10
#define ARR_SIZE (10)
#define MASK_SIZE (5)
#endif
int err;
float* createArray(int size)
{
    float * temp;
    temp = (float*)malloc(sizeof(float)*size);
    if( temp == NULL)
    {
        printf("Could not allocate memory :( \n");
    }
    assert(temp != NULL); 
    return temp;
}
void destroyArray(float* arr)
{
    free(arr);
}
void initArray(float * arr, int size)
{
    for( int i=0;i<size;i++)
    {
        int a = rand() % MAX_VAL;
        int b = rand() % MAX_VAL;
        if(DEBUG)
            arr[i] = (float)a;
        else
            arr[i] = (float)a/(b+1);
    }
}
void conv(float *inarr, float *mask, float *outarr, int arrsize, int masksize)
{
    int len = masksize/2;
    for( int i=0;  i< arrsize; i++)
    {
        float sum = 0;
        for( int j=0; j<masksize; j++)
        {
            if(i-len+j >=0 && i-len+j < arrsize);
            sum += mask[j]*inarr[i-len+j];
        }
        outarr[i] = sum;
    }
}
void printArray(float *arr, int size)
{
    for( int i=0; i<size; i++)
    {
        printf("%2.4f  ",arr[i]);
    }
}
int main()
{
    float *in, *out, *mask;
    int arrsize = ARR_SIZE;
    int masksize = MASK_SIZE;

    in = createArray(arrsize);
    out = createArray(arrsize);
    mask = createArray(masksize);

    initArray(in,arrsize);
    initArray(mask,masksize);

    conv(in, mask, out, arrsize, masksize);
    if(DEBUG)
    {
        printf("This is Input Array : ");
        printArray(in, arrsize);
        printf("\n");
        printf("This is Mask Array : ");
        printArray(mask, masksize);
        printf("\n");
        printf("This is Output Array : ");
        printArray(out, arrsize);
    } 

    free(in);
    free(out);
    free(mask);

    return 0;
}
