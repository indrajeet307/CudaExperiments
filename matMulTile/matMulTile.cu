/*
 * Tiled version of matrix multiplication
 * Sequential Matrix multiplication
 * TODO Step 1. Matrix dimensions are multiples of TILE_WIDTH [DONE]
 * TODO Step 1.a make each thread do more work
 * TODO Step 2. MAtrix dimensions are arbitary size
 */
#include<stdio.h>
#include<assert.h>
#include<cuda.h>
#include<stdlib.h>
#include<sys/time.h>
#include<errno.h>

#define DEBUG 0
#define VAL_LIMIT 10
 // TILE_WIDTH and MAT_DIM can be given at compile time check Makefile
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

#ifndef MAT_DIM
#define MAT_DIM 1024 
#endif

cudaError_t cuerr;

/*
*	@DESC   : Allocate memory a linear aare of dimension r*c 
*	@PRAM   : number of rows and columns
*	@RETURN : address of the allocated memory
*	@SEE    :  
*	@TODO   :  
*	
*/
float *createMartrix(int r, int c)
{
    float * temp;
    temp = (float*) malloc(sizeof(float) *r *c);
    if(temp == NULL)
        printf("Cannot create matrix :(\n");
    return temp;
}

/*
*	@DESC   : Free the linear array memory
*	@PRAM   : pointer to the array
*	@RETURN : nothing
*	@SEE    :  
*	@TODO   :  
*	
*/
void destroyMAtrix(float *m)
{
    free(m);
}

/*
*	@DESC   : Initialize matrix with some random values
*	@PRAM   : pointer to the matrix and its dimensions
*	@RETURN : nothing
*	@SEE    :  
*	@TODO   :  
*	
*/
void initMatrix(float *m, int r, int c)
{
    for( int i=0; i<r; i++)
    {
        for( int j=0; j<c; j++)
        {
            m[ i*c + j] = (float) (rand()%VAL_LIMIT);
        }
    }
}

/*
*	@DESC   : Sequential multiplication of matrix A and B result sotred in C
*	@PRAM   : host pointer to matrices A, B, and C dimensions of matrix C  and common
*           : dimension of matrix A, B
*	@RETURN : nothing
*	@SEE    :
*	@TODO   :
*	
*/
void matMul(float *A, float *B, float *C, int Ac, int Ar, int Bc) // Br == Ac
{
    for( int i=0; i<Ar; i++)
    {
        for( int j=0; j<Bc; j++)
        {
            float sum =0;
            for( int k=0; k<Ac; k++)
            {
                float a = A[ i*Ac + k];
                float b = B[ k*Bc + j];
                sum += a*b;
            }
            C[ i*Bc + j] = sum;
        }
    }
}

/*
 *	@PRAM   : Device pointer, number of rows and columns
 *	@RETURN : Nothing
 *	@DESC   : Creates a matrix of float * rows * columns on device
 *	@SEE    :  
 *	@TODO   :  
 *	
 */
void createMatrixDevice(float **m, int r, int c)
{
    int size = sizeof(float)*r*c;
    cuerr = cudaSuccess;
    cuerr = cudaMalloc(m, size); 
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "%s, %d.\n %s.", __FILE__, __LINE__, cudaGetErrorString(cuerr));
        exit(EXIT_FAILURE);
    }
}

/*
 *	@PRAM   : Host pointer, Device pointer, Number of rows and columns
 *	@RETURN : Nothing
 *	@DESC   : Copies data from host pointer to device pointer
 *	@SEE    :  
 *	@TODO   :  
 *	
 */
void transferToDevice(float *hostptr, float *deviceptr, int r, int c)
{
    int size = sizeof(float) * r*c;
    cuerr = cudaSuccess;
    cuerr = cudaMemcpy(deviceptr, hostptr, size, cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess)
    {
        //fprintf(stderr, "%s, %d.\n %s.", __FILE__, __LINE__, cudaGetErrorString(err));
        fprintf(stderr, "%s, %d.\n %s", __FILE__, __LINE__, cudaGetErrorString(cuerr));
        exit(EXIT_FAILURE);
    }
}

/*
 *	@PRAM   : Host pointer, Device pointer, Number of rows and columns
 *	@RETURN : Nothing
 *	@DESC   : Copies data from device pointer to host pointer
 *	@SEE    :  
 *	@TODO   :  
 *	
 */
void transferFromDevice(float *hostptr, float *deviceptr, int r, int c)
{
    int size = sizeof(float) * r*c;
    cuerr = cudaSuccess;
    cuerr = cudaMemcpy(hostptr, deviceptr, size, cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "%s, %d.\n %s.", __FILE__, __LINE__, cudaGetErrorString(cuerr));
        exit(EXIT_FAILURE);
    }
}

/*
*	@DESC   : Multiplies matrix A with matrix B and stores output in C
*	@PRAM   : device pointers matrix A, matrix B, matrix C, dimensions of matrix C and
*	comman dimension for matrix A and B
*	@RETURN : Nothing
*	@SEE    : Tiled matrix multiplication
*	@TODO   : A detailed description
*	
*/
__global__ 
void matMulKernel(float *A, float *B, float *C, int Ac, int Ar, int Bc)
{
    __shared__ float Sa[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Sb[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Crow = by * TILE_WIDTH + ty;
    int Ccol = bx * TILE_WIDTH + tx;
    float sum;

    for( int phase=0 ; phase < MAT_DIM/TILE_WIDTH; phase++)
    {
        Sa[ty][tx] = A[Crow*MAT_DIM + phase*TILE_WIDTH + tx];
        Sb[ty][tx] = B[ (phase*TILE_WIDTH + ty)*MAT_DIM + Ccol];
        __syncthreads();
        
        for( int i=0 ; i<TILE_WIDTH ; i++)
        {
            float a = Sa[ty][i];
            float b = Sb[i][tx];
            sum += a*b;
        }
        __syncthreads();
    }
    C[Crow*MAT_DIM +  Ccol] = sum;
}

/*
*	@DESC   : Wrapper function to set kernel configuration and invoke the kernel
*	@PRAM   : device pointers for matrix A, B, C, and dimensions for C
*	@RETURN : nothing
*	@SEE    :  
*	@TODO   :  
*	
*/
void pMatMul(float *A, float *B, float *C, int Ac, int Ar, int Bc)
{
    dim3 gridprop(ceil(Bc/TILE_WIDTH), ceil(Ar/TILE_WIDTH), 1);
    dim3 blockprop(TILE_WIDTH, TILE_WIDTH, 1);
    matMulKernel<<<gridprop, blockprop>>>(A, B, C, Ac, Ar, Bc);
}

/*
*	@DESC   : Print the matrix
*	@PRAM   : host pointer to the matrix and its dimensions
*	@RETURN : nothing   
*	@SEE    :  
*	@TODO   : 
*	
*/
void printMat(float *A, int r, int c)
{ for( int i=0; i<r; i++)
    {
        for( int j=0; j<c; j++)
        {
            printf("%3.2f\t", A[ i*c +j]);
        }
        printf("\n");
    }
}

/*
*	@DESC   : Check if the two given matrices are equal
*	@PRAM   : host matrix pointer A, B and their dimensions
*	@RETURN : true if matrices are equal else false
*	@SEE    :  
*	@TODO   :  
*	
*/
bool check(float *A, float *B, int r, int c)
{
    for( int i=0; i<r*c; i++)
    {
        if(A[i] != B[i])
            return false;
    }
    return true;
}

int main()
{
    float *h_A, *h_B, *h_C, *h_D;
    float *d_A, *d_B, *d_C;
    float milli;

    int Ar = MAT_DIM, Ac = MAT_DIM;
    int Br = MAT_DIM, Bc = MAT_DIM;
    int Cr = MAT_DIM, Cc = MAT_DIM;

    assert(Ac == Br); // Matrix are multipliable

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    h_A = createMartrix(Ar, Ac);
    assert(h_A != NULL);

    h_B = createMartrix(Br, Bc);
    assert(h_B != NULL);

    h_C = createMartrix(Cr, Cc);
    assert(h_C != NULL);

    h_D = createMartrix(Cr, Cc);
    assert(h_D != NULL);

    initMatrix(h_A, Ar, Ac);
    if(DEBUG)
    {
        printf("MAtrix A:\n");
        printMat(h_A, Ar, Ac);
    }

    initMatrix(h_B, Br, Bc);
    if(DEBUG)
    {
        printf("Matrix B:\n");
        printMat(h_B, Br, Bc);
    }

    matMul(h_A, h_B, h_C, Ac, Ar, Bc);

    if(DEBUG)
    {
        printf("Matrix C:\n");
        printMat(h_C, Cr, Cc);
    }

    createMatrixDevice(&d_A, Ar, Ac);
    createMatrixDevice(&d_B, Br, Bc);
    createMatrixDevice(&d_C, Cr, Cc);

    transferToDevice(h_A, d_A, Ar, Ac);
    transferToDevice(h_B, d_B, Br, Bc);

    cudaEventRecord(start);
    pMatMul(d_A, d_B, d_C, Ac, Ar, Bc);
    cudaEventRecord(stop);

    transferFromDevice(h_D, d_C, Cr, Cc);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);
    printf("Time required for (Configuration %d TILE_WIDTH and %d MAT_DIM) parallel \
    execution %f\n", TILE_WIDTH, MAT_DIM, milli);
    #if defined(TILE_WIDTH) && defined(MAT_DIM)
        char cmd[1024];
        char vals[256];
        sprintf(vals, "%d\t%d\t%f", TILE_WIDTH, MAT_DIM, milli);
        strcpy(cmd, "echo \"");
        strcat(cmd, vals);
        strcat(cmd, "\" >>res.data");
        system(cmd);
    #endif
    if(DEBUG)
    {
        printf("Matrix C:\n");
        printMat(h_C, Cr, Cc);
    }

    if(check(h_D, h_C, Cr, Cc))
    {
        printf("Success :) \n");
    }
    else
    {
        printf("Failed :( \n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
