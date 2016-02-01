/*
   Code for 1d convolution
   Code contains use of constant memory
 */
#include<stdio.h>
#include<cuda.h>
#include<errno.h>
#include<assert.h>
#include<sys/time.h>
#define DEBUG 0
#if not defined(DEBUG)
#define MAX_VAL 800
#define ARR_SIZE (4096*4096*3)
#define MASK_SIZE 11
#else
#define MAX_VAL 10
#define ARR_SIZE (4096*4096)
#define MASK_SIZE (111)
#endif
#define BLOCK_WIDTH 512
cudaError_t err;
__constant__ float cuMASK[MASK_SIZE];

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
        arr[i] = (float)a;
    }
}

void createArrayDevice(float **arr, int size)
{
    err = cudaSuccess;
    err = cudaMalloc(arr,size*sizeof(float));
    if(err != cudaSuccess)
    {
        fprintf(stderr, "#Error %s, %d.\n%s.",__FILE__,__LINE__,cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void transferToDevice(float *hostptr, float *deviceptr, int size)
{
    err = cudaSuccess;
    err = cudaMemcpy(deviceptr, hostptr, sizeof(float)*size,cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "#Error %s, %d.\n%s.", __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void transferFromDevice(float *hostptr, float *deviceptr, int size)
{
    err = cudaSuccess;
    err = cudaMemcpy(hostptr, deviceptr, sizeof(float)*size, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "#Error %s, %d.\n%s.", __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
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

    __global__
void convKernel(float *inarr, float *outarr, int arrsize, int masksize)
{
    int in = blockDim.x * blockIdx.x + threadIdx.x;
    if( in < arrsize)
    {
        int len = masksize/2;
        float sum = 0;
        for( int i=0; i<masksize; i++)
        {
            if( in-len+i >=0 && in-len+i < arrsize)
            {
                sum += cuMASK[i] * inarr[in-len+i];
            }
        }
        outarr[in] = sum;
    }
}

void pconv(float *inarr, float *outarr, int arrsize, int masksize)
{
    dim3 gridProp(ceil(arrsize/BLOCK_WIDTH), 1, 1);
    //dim3 gridProp(1, 1, 1);
    dim3 blockProp(BLOCK_WIDTH, 1, 1);
    err = cudaSuccess;
    convKernel<<<gridProp,blockProp>>>(inarr, outarr, arrsize, masksize);
    if (err != cudaSuccess)
    {
        fprintf(stderr,"%s, %d.\n %s.",__FILE__,__LINE__,cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void printArray(float *arr, int size)
{
    for( int i=0; i<size; i++)
    {
        printf("%2.4f  ",arr[i]);
    }
}

bool check(float *arr, float *arr2, int size)
{
    for( int i=0; i<size; i++)
    {
        if( arr[i] != arr2[i])
        {
            printf("%f != %f @ [%d]\n", arr[i], arr2[i], i);
            return false;
        }
    }
    return true;
}

int main()
{
    float *in, *out, *mask, *h_out;
    float *d_in, *d_out;
    
    int arrsize = ARR_SIZE;
    int masksize = MASK_SIZE;
    cudaEvent_t start, stop;
    timeval startseq, stopseq, diffseq;
    float milli;

    cudaEventCreate(&start);
cudaEventCreate(&stop);


    in = createArray(arrsize);
    out = createArray(arrsize);
    h_out = createArray(arrsize);
    mask = createArray(masksize);

    createArrayDevice(&d_in, arrsize);
    createArrayDevice(&d_out, arrsize);

    initArray(in,arrsize);
    initArray(h_out,arrsize);
    initArray(mask,masksize);

    transferToDevice(in, d_in, arrsize);
    transferToDevice(h_out, d_out, arrsize);
    err = cudaSuccess;
    err = cudaMemcpyToSymbol(cuMASK,mask, sizeof(float)*masksize);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "#Error %s, %d.\n%s.", __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    gettimeofday(&startseq,NULL);
    conv(in, mask, out, arrsize, masksize);
    gettimeofday(&stopseq,NULL);

    timersub(&stopseq,&startseq,&diffseq);
    printf("Time required for (Configuration %d ARR_SIZE and %d BLOCK_WIDTH) sequential \
execution %f\n",ARR_SIZE,BLOCK_WIDTH,(float)(diffseq.tv_sec*1000)+(diffseq.tv_usec/1000));
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
    cudaEventRecord(start);
    pconv(d_in, d_out, arrsize, masksize);
    cudaEventRecord(stop);

    transferFromDevice(h_out, d_out, arrsize);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milli, start, stop);
    printf("Time required for (Configuration %d ARR_SIZE and %d BLOCK_WIDTH) parallel \
execution %f\n",ARR_SIZE,BLOCK_WIDTH,milli);

    if(check(out, h_out, arrsize))
        printf("Yes\n");
    else
        printf("No\n");

    cudaFree(d_in);
    cudaFree(d_out);

    free(h_out);
    free(in);
    free(out);
    free(mask);

    return 0;
}
