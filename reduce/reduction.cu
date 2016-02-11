#include<stdio.h>
#include<assert.h>
#include<cuda.h>
#include<errno.h>
#include<math.h>
#include<sys/time.h>
#define MAX_VAL 10
#define BLOCK_WIDTH 256
#define MAX_SIZE 2048*2048*2
cudaError_t cuerr;
float* createArray(int size)
{
    float *temp;
    int err=0;
    errno = 0;
    temp = (float*) malloc (sizeof(float)*size);
    err = errno;
    if(err)
        printf("Error: %s, %d, %s", __FILE__, __LINE__, strerror(err));
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
}

void reduce(float* p, int size)
{
    for( size_t j=1;j<=size/2;j*=2)
    {
        for( size_t i=0;i<size/2;i++)
        {
            size_t ind = 2*i*j;
            if(ind < size)
                p[ind] = p[ind]+p[ind+j];
        }
        printArray(p,10);
    }
}

void reduce2(float* p, int size)
{
    for( size_t j=size/2;j>0;j/=2)
    {
        for( size_t i=0;i<size/2;i++)
        {
            if(i+j < size)
                p[i] = p[i]+p[i+j];
        }
    }
}

void createArrayDevice(float **p, int size)
{
    cuerr = cudaSuccess;
    cuerr = cudaMalloc(p, sizeof(float)*size);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "%s, %d.\n %s.", __FILE__, __LINE__, cudaGetErrorString(cuerr));
        exit(EXIT_FAILURE);
    }
}

void transferToDevice(float *hostptr, float *deviceptr, int size)
{
    cuerr = cudaSuccess;
    cuerr = cudaMemcpy(deviceptr, hostptr, sizeof(float)*size, cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "%s, %d.\n %s.", __FILE__, __LINE__, cudaGetErrorString(cuerr));
        exit(EXIT_FAILURE);
    }
}

void transferFromDevice(float *hostptr, float *deviceptr, int size)
{
    cuerr = cudaSuccess;
    cuerr = cudaMemcpy(hostptr, deviceptr, sizeof(float)*size, cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "%s, %d.\n %s.", __FILE__, __LINE__, cudaGetErrorString(cuerr));
        exit(EXIT_FAILURE);
    }
}

    __global__
void reduceKernel(float *p, int size, float *con)
{
    __shared__ float localblock[BLOCK_WIDTH*2];
    int tx = threadIdx.x;
    int in = blockIdx.x * blockDim.x + tx;
    localblock[tx*2] = p[in*2];
    localblock[tx*2+1] = p[in*2+1];
    __syncthreads();

    for( int it=1; it<= (BLOCK_WIDTH*2)/2; it*=2)
    {
        int ind = 2 * tx * it;
        if(ind < blockDim.x*2)
            localblock[ind] = localblock[ind]+localblock[ind+it];
        __syncthreads();
    }
    p[in]=localblock[tx];
    if(tx == 0) con[blockIdx.x] = localblock[tx];
}

void preduce(float *p, int size, float *con)
{
    //dim3 gridProp(ceil(size/(BLOCK_WIDTH*2)),1,1);
    dim3 gridProp((int)ceil(size/(BLOCK_WIDTH*2)),1,1);
    dim3 blockProp(BLOCK_WIDTH,1,1);
    printf("\nRunnig Kernel with %d thpb, %d bpg\n", BLOCK_WIDTH, (int)ceil(size/(BLOCK_WIDTH*2)));
    cuerr = cudaSuccess;
    reduceKernel<<<gridProp,blockProp>>>(p, size, con);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "%s, %d.\n %s.", __FILE__, __LINE__, cudaGetErrorString(cuerr));
        exit(EXIT_FAILURE);
    }
}

    __global__
void reduceKernel2(float *p, int size)
{
    __shared__ float localblock[BLOCK_WIDTH*2];
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    localblock[tx*2] = p[tx*2];
    localblock[tx*2+1] = p[tx*2+1];
    __syncthreads();

    for( int it=size/2; it>0; it/=2)
    {
        if(tx+it < blockDim.x*2)
            localblock[tx] = localblock[tx]+localblock[tx+it];
        __syncthreads();
    }
    p[tx] = localblock[tx]; 
}

void preduce2(float *p, int size)
{
    //dim3 gridProp(ceil(size/(BLOCK_WIDTH*2)),1,1);
    dim3 gridProp(1,1,1);
    dim3 blockProp(BLOCK_WIDTH,1,1);
    printf("Runnig Kernel with %d thpb, %d bpg\n", BLOCK_WIDTH, (int)ceil(size/(BLOCK_WIDTH*2)));
    cuerr = cudaSuccess;
    reduceKernel2<<<gridProp,blockProp>>>(p, size);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "%s, %d.\n %s.", __FILE__, __LINE__, cudaGetErrorString(cuerr));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    float *A,*B,*C;
    int Asize = MAX_SIZE;
    struct timeval as,ae,bs,be,ad,bd; 
    cudaEvent_t custart, cuend;
    float timeelap;
    cudaEventCreate(&custart);
    cudaEventCreate(&cuend);

    float *container;
    int csize = (MAX_SIZE/(BLOCK_WIDTH*2));

        
    A = createArray(Asize);
    assert(A != NULL);
    B = createArray(Asize);
    assert(B != NULL);
    C = createArray(Asize);
    assert(C != NULL);
    container = createArray(csize);
    assert(container != NULL);

    for( int i=0;i<csize;i++)
        container[i] = 0; 

    initArray(A, Asize);
    copyArray(A, B, Asize);
    copyArray(A, C, Asize); 

    gettimeofday(&as,NULL);
    reduce(A, Asize);
    gettimeofday(&ae,NULL);
    //printf("\nA[0] is %f\n",A[0]);
    //printArray(A, Asize);

    gettimeofday(&bs,NULL);
    reduce2(B, Asize);
    gettimeofday(&be,NULL);
    //printf("\nB[0] is %f\n",B[0]);

    float *dA, *dB, *dcont;
    createArrayDevice(&dA, Asize);
    createArrayDevice(&dcont, csize);

    printArray(C,10);
    cudaEventRecord(custart);
    transferToDevice(C, dA, Asize);
    preduce(dA, Asize, dcont);
    cudaEventRecord(cuend);

    cudaEventSynchronize(cuend);
    cudaDeviceSynchronize();

    cudaEventElapsedTime(&timeelap, custart, cuend);
    transferFromDevice(C, dA, Asize);
    transferFromDevice(container, dcont, csize);

    printf("\n[");
    float finsum=0;
    for( int i=0;i<csize;i++)
        finsum += (float)container[i];
        printf("%f", finsum);
    printf("]\n");
    printArray(C,10);
    printf("\nC[0] is %f\n",C[0]);

    destroyArray(A);
    destroyArray(B);

    timersub(&ae, &as, &ad);
    timersub(&be, &bs, &bd);

    printf(" %f reduce\n", (float)(ad.tv_sec*1000)+(ad.tv_usec/1000));
    printf(" %f reduce2\n",(float)(bd.tv_sec*1000)+(bd.tv_usec/1000));
    printf(" %f preduce2\n", timeelap);
    return 0;
}
/*
   If the array size spans multiple blocks
   create MAX_SIZE/BLOCK_WIDTH array
   This array will store value from each block
   rerun reduction on this array or use CPU to calcualte the final value 
 */
