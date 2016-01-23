#include<stdio.h>
#include<cuda.h> 
#include<stdlib.h> 
#define VALUES 1000
cudaError_t err=cudaSuccess;

void vecAdd(float *h_A, float *h_B, float *h_C,int size)
{
    for(int i=0; i<size; i++)
    {
        h_C[i] = h_A[i] + h_B[i];
    }
}

__global__ 
void vecAddKernel(float *A, float *B, float *C,int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i<size ) C[i] = B[i] + A[i];
}

__host__
void pvecAdd(float *h_A, float *h_B, float *h_C,int size)
{
    int size_in_bytes = size * sizeof(float);
    float *d_A, *d_B, *d_C;

    err = cudaMalloc((void**) &d_A, size_in_bytes);
    if (err != cudaSuccess)
    {
        fprintf(stderr,"%s\n.",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy( d_A, h_A, size_in_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr,"%s\n.",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc((void**) &d_B, size_in_bytes);
    if (err != cudaSuccess)
    {
        fprintf(stderr,"%s\n.",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy( d_B, h_B, size_in_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr,"%s\n.",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**) &d_C, size_in_bytes);
    if (err != cudaSuccess)
    {
        fprintf(stderr,"%s\n.",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //dim3 dim_grid((size -1)/256+1 , 1, 1);
    //dim3 dim_block(256, 1, 1);
    int threadPB = 256;
    int blocksPG = (size + threadPB -1)/threadPB;
    vecAddKernel<<<blocksPG, threadPB>>>(d_A, d_B, d_C, size);

    err = cudaMemcpy( h_C, d_C, size_in_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr,"%s\n.",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int check(float *h_A, float *h_B, int size)
{
    for(int i=0;i<size;i++)
    {
        if ( h_A[i] != h_B[i] ) {
        printf(" %f != %f\t", h_A[i], h_B[i]);
        }
    }
    return 1;
}

void initData(float *h_A, float *h_B, int size)
{
    for(int i=0;i<size;i++)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }
}

int main()
{
    float *h_A,*h_B,*h_C,*h_D;
    unsigned int size = VALUES;

    h_A = (float*) malloc (sizeof(float)* size);
    h_B = (float*) malloc (sizeof(float)* size);
    h_C = (float*) malloc (sizeof(float)* size);
    h_D = (float*) malloc (sizeof(float)* size);

    initData(h_A, h_B, size);
    vecAdd(h_A, h_B, h_C, size);
    pvecAdd(h_A, h_B, h_D, size);
        
    if(check(h_C, h_D, size))
        printf("Success !\n");
    else
        printf("UnSuccess !\n");
    return 0;
}
