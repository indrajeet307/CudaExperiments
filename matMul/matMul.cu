/*
    Program perform matrix multiplication
*/
#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>
#define VAL_LIMIT 10
#define DEBUG 0
#define TILE_WIDTH 2
cudaError_t err;

float* createMatrix(int r,int c)
{
    float *temp;
    temp = (float*) malloc(sizeof(float)*r*c);
    return temp;
}

void createMatrixDevice(float **m, int r, int c)
{
    int size = sizeof(float)*r*c;
    err = cudaSuccess;
    err = cudaMalloc(m, size); 
    if (err != cudaSuccess)
    {
        fprintf(stderr,"%s, %d.\n %s.",__FILE__,__LINE__,cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void transferToDevice(float *hostptr, float *deviceptr, int r, int c)
{
    int size = sizeof(float) * r*c;
    err = cudaSuccess;
    err = cudaMemcpy(deviceptr,hostptr,size,cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        //fprintf(stderr,"%s, %d.\n %s.",__FILE__,__LINE__,cudaGetErrorString(err));
        fprintf(stderr,"%s",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void transferFromDevice(float *hostptr, float *deviceptr, int r, int c)
{
        int size = sizeof(float) * r*c;
        err = cudaSuccess;
        err = cudaMemcpy(hostptr,deviceptr,size,cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr,"%s, %d.\n %s.",__FILE__,__LINE__,cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
}

void initMatrix(float *m,int r,int c)
{
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            m[ i*c +j ] = (float) (rand()%VAL_LIMIT);
        }
    }
}

void matMul(float *A, float *B, float *C, int Ac, int Ar, int Br)
{
    for( int i=0 ; i<Ar; i++)
    {
        for( int j=0; j<Ac; j++)
        {
            float sum=0;
            for( int k=0; k<Br; k++) 
            {
                float a = A[i*Ac +k];
                float b = B[k*Ar +j];
                sum += a*b;
            }
            C[i*Ac+j] = sum;
        }
    }

}

__global__
void matMulKernel(float *A, float *B, float *C, int Ac, int Ar, int Br)
{
    int row = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int col = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int sum = 0;
    for ( int i=0 ; i<Ar; i++)
    {
        sum += A[row *Ac + i] * B[ i *Br + col];
    }
    C[row*Ar+col] = sum;
}

void pMatMul(float *A,float *B,float *C, int Ac, int Ar, int Br)
{
    dim3 gridProp(ceil(Ac/TILE_WIDTH),ceil(Ar/TILE_WIDTH),1);
    dim3 blockProp(TILE_WIDTH,TILE_WIDTH,1);
    matMulKernel<<<gridProp,blockProp>>>(A, B, C, Ac, Ar, Br);
}


void printMat(float *mat, int r, int c)
{
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            printf("%4.1f \t",mat[i*r+j]);
        }
        printf("\n");
    }
}

bool check(float *mat, float *mat2, int r, int c)
{
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            if( mat2[i*r+j] != mat[i*r+j])
            return false;
        }
    }
    return true;
}

int main()
{
    float *h_A, *h_B, *h_C,*h_D;
    float *d_A, *d_B, *d_C;
    unsigned int Ar=512, Ac=1024;
    unsigned int Br=1024, Bc=512;
    unsigned int Cr=512, Cc=512;

    h_A = createMatrix(Ar, Ac);
    h_B = createMatrix(Br, Bc);
    h_C = createMatrix(Cr, Cc);
    h_D = createMatrix(Cr, Cc);
    
    initMatrix(h_A, Ar, Ac);
    initMatrix(h_B, Br, Bc);

    if(DEBUG){ 
        printf("Matrix A:\n");
        printMat(h_A, Ar, Ac);
        printf("Matrix B:\n");
        printMat(h_B, Br, Bc);
    }


    matMul(h_A, h_B, h_C, Ac, Ar, Br);

    if(DEBUG){ 
        printf("Matrix C:\n");
        printMat(h_C, Cr, Cc);
    }


    createMatrixDevice(&d_A, Ar, Ac);
    createMatrixDevice(&d_B, Br, Bc);
    createMatrixDevice(&d_C, Cr, Cc);

    transferToDevice(h_A, d_A, Ar, Ac);
    transferToDevice(h_B, d_B, Br, Bc);

    pMatMul(d_A, d_B, d_C, Ac, Ar, Br);

    transferFromDevice(h_D, d_C, Cr, Cc);

    if(DEBUG){
        printf("Matrix C:\n");
        printMat(h_D, Cr, Cc);
    }

    if(check(h_D, h_C, Cr, Cc))
        printf("Success !! :) \n");
    else
        printf("Failed !! :( \n");

    return 0;
}
