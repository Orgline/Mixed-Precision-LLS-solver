#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

#include <mkl.h>

#include <cuda_fp16.h>
#include <cusolverDn.h>

#include <basicop.cuh>

#include <singleLLS.cu>
#include <doubleLLS.cu>
#include <recursiveQR.cu>

//number of rows and columns of matrix A
int nrA, ncA;

//read from file if mode = 1, generate on GPU if mode = 2
int mode;

int  parseArguments(int argc,char *argv[])
{
    if(argc < 4)
    {
        printf("Need at least 4 inputs\n");
        return -1;
    }
    nrA = atoi(argv[1]);
    ncA = atoi(argv[2]);
    mode = atoi(argv[3]);
    return 0;
}

int main(int argc,char *argv[])
{
    if(parseArguments(argc, argv)==-1)
    {
        //printf("Run RGEQRF\n");
        return 0;
    }
    float* hA;
    hA = (float*)malloc(sizeof(float)*nrA*ncA);
    if(mode == 1)
    {
        printf("Run RGEQRF\n");
        float *dA;
        cudaMalloc(&dA,sizeof(float)*nrA*ncA);
        generateMatrixDevice(dA,nrA,ncA);
        cudaMemcpy(hA,dA,sizeof(float)*nrA*ncA,cudaMemcpyDeviceToHost);
        cudaFree(dA);
        recursiveQR(nrA,ncA,hA,nrA);
    }
    if(mode == 2)
    {
        float *dA;
        cudaMalloc(&dA,sizeof(float)*nrA*ncA);
        generateMatrixDevice(dA,nrA,ncA);
        cudaMemcpy(hA,dA,sizeof(float)*nrA*ncA,cudaMemcpyDeviceToHost);
        cudaFree(dA);
        sSolveLLS(nrA,ncA,hA,nrA);
    }
    if(mode == 3)
    {
        double *dA;
        cudaMalloc(&dA,sizeof(double)*nrA*ncA);
        generateMatrixDevice(dA,nrA,ncA);

        double* doubleA;
        doubleA = (double*)malloc(sizeof(double)*nrA*ncA);
        cudaMemcpy(doubleA,dA,sizeof(doubleA)*nrA*ncA,cudaMemcpyDeviceToHost);
        cudaFree(dA);
        
        
        dSolveLLS(nrA,ncA,doubleA,nrA);
    }

    return 0;
}