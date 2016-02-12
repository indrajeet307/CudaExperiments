#include<stdio.h>
#include<assert.h>
#include<cuda.h>
#include<errno.h>
#include<math.h>
#include<sys/time.h>
#define MAX_VAL 100
#ifndef BLOCK_WIDTH
#define BLOCK_WIDTH 256
#endif
#ifndef MAX_SIZE
#define MAX_SIZE 512 // working on single block 
#endif
#define TILE_WIDTH (BLOCK_WIDTH*2)
cudaError_t cuerr;
float* createArray(int size)
{
    float *temp;
    int err=0;
    errno = 0;
    temp = (float*) malloc (sizeof(float)*size);
    err = errno;
    if(err)
        printf("Error: %s, %d, %s\n", __FILE__, __LINE__, strerror(err));
    return temp;
}

void destroyArray(float* p)
{
    free(p);
}

void initArray(float* p, int size)
{
    for(int i=0; i<size; i++)
        p[i] =  rand()%MAX_VAL;
}

void copyArray(float *p, float *q, int size)
{
    for(int i=0;i<size;i++)
        q[i]=p[i];
}

void printArray(float *p, int size)
{
    printf("\n");
    for( int i=0; i< size; i++)
        printf("%4.2f ",p[i]);
    printf("\n");
}

void prefixSum(float* p, float *q, int size)
{
    q[0] = p[0];
    for( int i=1; i< size;i++)
        q[i] = q[i-1] + p[i];
}

void createArrayDevice(float **p, int size)
{
    cuerr = cudaSuccess;
    cuerr = cudaMalloc(p, sizeof(float)*size);
    if (cuerr != cudaSuccess)
    {
        printf( "%s, %d.\n %s.\n", __FILE__, __LINE__, cudaGetErrorString(cuerr));
        exit(EXIT_FAILURE);
    }
}

void transferToDevice(float *hostptr, float *deviceptr, int size)
{
    cuerr = cudaSuccess;
    cuerr = cudaMemcpy(deviceptr, hostptr, sizeof(float)*size, cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess)
    {
        printf("%s, %d.\n %s.", __FILE__, __LINE__, cudaGetErrorString(cuerr));
        exit(EXIT_FAILURE);
    }
}

void transferFromDevice(float *hostptr, float *deviceptr, int size)
{
    cuerr = cudaSuccess;
    cuerr = cudaMemcpy(hostptr, deviceptr, sizeof(float)*size, cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess)
    {
        printf( "%s, %d.\n %s.", __FILE__, __LINE__, cudaGetErrorString(cuerr));
        exit(EXIT_FAILURE);
    }
}

    __global__
void preFixSumKernel(float *p, float *q, int size)
{
    __shared__ float localblock[TILE_WIDTH];
    int tx = threadIdx.x;
    int in = blockIdx.x * blockDim.x + tx;

    // each thread will load two elements
    localblock[tx*2] = p[in*2];
    localblock[tx*2+1] = p[in*2+1];
    __syncthreads();

    for( int stride=1; stride<(TILE_WIDTH); stride*=2)
    {
         __syncthreads();
         int index = (tx+1) * (stride * 2) - 1;
         if( index < TILE_WIDTH)
              localblock[index] += localblock[index-stride];
    }

    for ( int stride=(TILE_WIDTH)/4 ; stride > 0 ; stride/=2)
    {
         __syncthreads();
         int index = (tx+1) * (stride*2) -1;
         if( (index + stride )< TILE_WIDTH)
              localblock[index +stride] += localblock[index];
    }
   q[in*2]   = localblock[tx*2];
   q[in*2+1] = localblock[tx*2+1]; 
}

void pprefixSum(float *p, float *q, int size)
{
    //dim3 gridProp((int)ceil(size/(TILE_WIDTH)),1,1);
    dim3 gridProp(1, 1, 1);
    dim3 blockProp(BLOCK_WIDTH,1,1);
    printf("\nRunnig Kernel with %d thpb, %d bpg\n", BLOCK_WIDTH,
    (int)ceil(size/(TILE_WIDTH)));
    cuerr = cudaSuccess;
    preFixSumKernel<<<gridProp,blockProp>>>(p, q, size);
    if (cuerr != cudaSuccess)
    {
        printf("%s, %d.\n %s.", __FILE__, __LINE__, cudaGetErrorString(cuerr));
        exit(EXIT_FAILURE);
    }
}

bool check ( float *a, float *b, int size)
{
    for ( int i=0; i<size; i++)
        if( a[i] != b[i])
        {
             printf(" %f != %f @%d\n", a[i] , b[i], i );
            return false;
        }
        else
             printf(" %f != %f @%d\n", a[i] , b[i], i );
    return true;
}

int main()
{
    float *A,*Aout;
    int Asize = MAX_SIZE;
    struct timeval as,ae,ad; 
    cudaEvent_t custart, cuend;
    float timeelap;
    cudaEventCreate(&custart);
    cudaEventCreate(&cuend);
        
    A = createArray(Asize);
    assert(A != NULL);
    Aout = createArray(Asize);
    assert(Aout != NULL);


    initArray(A, Asize);

    gettimeofday(&as,NULL);
    prefixSum(A, Aout, Asize);
    gettimeofday(&ae,NULL);
    //printf("\nA[0] is %f\n",A[0]);
    printArray(A, Asize);
    printArray(Aout, Asize);


    float *dA, *dAout, *dout;
    createArrayDevice(&dA, Asize);
    createArrayDevice(&dAout, Asize);
    dout = createArray(Asize);

    
    //printArray(C,10);
    cudaEventRecord(custart);
    transferToDevice(A, dA, Asize);
    pprefixSum(dA, dAout, Asize); 
    cudaEventRecord(cuend);

    cudaEventSynchronize(cuend);

    cudaEventElapsedTime(&timeelap, custart, cuend);
    transferFromDevice(dout, dAout, Asize);
    cudaDeviceSynchronize();

    if( check(Aout, dout, Asize) )
        printf("Correct Results\n");
    else
        printf("Something went wrong somewhere\n");

    timersub(&ae, &as, &ad);
    printf(" Serial implementation took %ld milisecs.\n", ad.tv_sec*1000+ad.tv_usec/1000);

    cudaFree(dA);
    cudaFree(dAout);

    destroyArray(A);
    destroyArray(Aout);
    destroyArray(dout);
    return 0;
}
