/*
   Program perform matrix multiplication
 */
#include<stdio.h>
#include<cuda.h>
#include<assert.h>
#include<stdlib.h>
#include<sys/time.h>
#define VAL_LIMIT 10
#define DEBUG 0
#define TILE_WIDTH 32
cudaError_t err;
/*
 *	@PRAM   : Number of rows and columns
 *	@RETURN : Pointer to created Matrix
 *	@DESC   :  
 *	@SEE    :  
 *	@TODO   : 
 *	
 */
float* createMatrix(int r,int c)
{
    float *temp;
    temp = (float*) malloc(sizeof(float)*r*c);
    return temp;
}

/*
 *	@DESC   : Frees the memory allocated to the matrix
 *	@PRAM   : pointer to the matrix
 *	@RETURN : Nothing
 *	@SEE    :  
 *	@TODO   :  
 *	
 */
void destroyMAtrix(float *mat)
{
    free(mat);
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
    err = cudaSuccess;
    err = cudaMalloc(m, size); 
    if (err != cudaSuccess)
    {
        fprintf(stderr,"%s, %d.\n %s.",__FILE__,__LINE__,cudaGetErrorString(err));
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

void matMul(float *A, float *B, float *C, int Aw, int Ah, int Bw)
{
    for( int i=0 ; i<Ah; i++)
    {
        for( int j=0; j<Bw; j++)
        {
            float sum=0;
            for( int k=0; k<Aw; k++) 
            {
                float a = A[i*Aw+k];
                float b = B[k*Bw +j];
                sum += a*b;
                if(DEBUG)
                    printf(" %d * %d +",i*Aw+k,k*Bw+j);
            }
            C[i*Bw+j] = sum;
                if(DEBUG)
                    printf("%d\n",i*Bw+j);
        }
    }

}

 __global__
void matMulKernel(float *A, float *B, float *C, int Ac, int Ar, int Bc)
{
    int row = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int col = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int sum = 0;
    for ( int i=0 ; i<Ar; i++)
    {
        sum += A[row *Ac + i] * B[ i *Bc + col];
    }
    C[row*Bc+col] = sum;
}

void pMatMul(float *A,float *B,float *C, int Ac, int Ar, int Bw)
{
    dim3 gridProp(ceil(Ac/TILE_WIDTH),ceil(Ar/TILE_WIDTH),1);
    dim3 blockProp(TILE_WIDTH,TILE_WIDTH,1);
    matMulKernel<<<gridProp,blockProp>>>(A, B, C, Ac, Ar, Bw);
}


void printMat(float *mat, int r, int c)
{
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            printf("%4.1f \t",mat[i*c+j]);
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
            if( mat2[i*c+j] != mat[i*c+j])
                return false;
        }
    }
    return true;
}

int main()
{
    float *h_A, *h_B, *h_C,*h_D;
    float *d_A, *d_B, *d_C;
    unsigned int Ar=1024, Ac=1024;
    unsigned int Br=1024, Bc=1024;
    unsigned int Cr=1024, Cc=1024;

    assert(Ac == Br);

    h_A = createMatrix(Ar, Ac);
    assert(h_A != NULL);
    h_B = createMatrix(Br, Bc);
    assert(h_B != NULL);
    h_C = createMatrix(Cr, Cc);
    assert(h_C != NULL);
    h_D = createMatrix(Cr, Cc);
    assert(h_D != NULL);

    initMatrix(h_A, Ar, Ac);
    initMatrix(h_B, Br, Bc);

    if(DEBUG){ 
        printf("Matrix A:\n");
        printMat(h_A, Ar, Ac);
        printf("Matrix B:\n");
        printMat(h_B, Br, Bc);
    }


    matMul(h_A, h_B, h_C, Ac, Ar, Bc);

    if(DEBUG){ 
        printf("Matrix C:\n");
        printMat(h_C, Cr, Cc);
    }


    createMatrixDevice(&d_A, Ar, Ac);
    createMatrixDevice(&d_B, Br, Bc);
    createMatrixDevice(&d_C, Cr, Cc);

    transferToDevice(h_A, d_A, Ar, Ac);
    transferToDevice(h_B, d_B, Br, Bc);
    struct timeval st, et, dt;
    gettimeofday(&st,NULL);
    pMatMul(d_A, d_B, d_C, Ac, Ar, Bc);
    gettimeofday(&et,NULL);

    timersub(&et, &st, &dt);
    printf("Time required %lf\n",dt.tv_sec * 1000.0 + dt.tv_usec/1000.0);
    transferFromDevice(h_D, d_C, Cr, Cc);


    if(DEBUG){
        printf("Matrix D:\n");
        printMat(h_D, Cr, Cc);
    }

    if(check(h_D, h_C, Cr, Cc))
        printf("Success !! :) \n");
    else
        printf("Failed !! :( \n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    destroyMAtrix(h_A);
    destroyMAtrix(h_B);
    destroyMAtrix(h_D);
    destroyMAtrix(h_C);
    return 0;
}
