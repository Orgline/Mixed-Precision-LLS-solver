#include <assert.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include<stdio.h>
#include<stdlib.h>


void printMatrixFloat(char *filename, int m, int n, float* a, int lda)
{

	FILE *f = fopen(filename, "w");
	if (f == NULL) {
		printf("fault!\n");
		return;
	}
	for (int i = 0; i < m; i++) {
		//printf("i = %d\n", i);
		for (int j = 0; j < n; j++) {
			fprintf(f, "%f", a[i + j*lda]);
			if (j == n - 1) fprintf(f, "\n");
			else fprintf(f, ",");
		}
	}
	fclose(f);
}

void printMatrixDouble(char *filename, int m, int n, double* a, int lda)
{

	FILE *f = fopen(filename, "w");
	if (f == NULL) {
		printf("fault!\n");
		return;
	}
	for (int i = 0; i < m; i++) {
		//printf("i = %d\n", i);
		for (int j = 0; j < n; j++) {
			fprintf(f, "%lf", a[i + j*lda]);
			if (j == n - 1) fprintf(f, "\n");
			else fprintf(f, ",");
		}
	}
	fclose(f);
}

void printMatrixDevice(char *filename,int m, int n, float *dA, int lda)
{
    //printf("Perform printmatrixdevice\n");
    float *ha;
    ha = (float*)malloc(sizeof(float)*m*n);
	cudaMemcpy(ha, dA, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    printMatrixFloat(filename, m, n, ha, lda);
    free(ha);
}

void printMatrixDevice(char *filename,int m, int n, double *dA, int lda)
{
    //printf("Perform printmatrixdevice\n");
    double *ha;
    ha = (double*)malloc(sizeof(double)*m*n);
	cudaMemcpy(ha, dA, sizeof(double)*m*n, cudaMemcpyDeviceToHost);
    printMatrixDouble(filename, m, n, ha, lda);
    free(ha);

}

__global__
void s2d(int m, int n, float *as, int ldas, double *ad, int ldad)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		ad[i + j*ldad] = (double)(as[i + j*ldas]);
	}
}

__global__
void s2h(int m, int n, float *as, int ldas, __half *ah, int ldah)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		ah[i + j*ldah] = __float2half(as[i + j*ldas]);
	}
}

__global__
void h2s(int m, int n,__half *ah, int ldah, float *as, int ldas)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		as[i + j*ldah] = __half2float(ah[i + j*ldas]);
	}
}

__global__
void s2hStoreResidule(int m, int n, float *as, int ldas, __half *ah, int ldah,__half *ar, int ldar)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
        ah[i + j*ldah] = __float2half(as[i + j*ldas]);
        ar[i + j*ldah] = __float2half(as[i + j*ldas]-__half2float(ah[i + j*ldah]));
	}
}

__global__
void s2hStoreResidule(int m, int n, float *as, int ldas, __half *ah, int ldah, float *ar, int ldar)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
        ah[i + j*ldah] = __float2half(as[i + j*ldas]);
        ar[i + j*ldah] = as[i + j*ldas]-__half2float(ah[i + j*ldah]);
	}
}

__global__
void s2hStoreResiduleZoom(int m, int n, float *as, int ldas, __half *ah, int ldah,__half *ar, int ldar,float mul)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
        ah[i + j*ldah] = __float2half(as[i + j*ldas]/mul);
        ar[i + j*ldah] = __float2half(as[i + j*ldas]/mul-__half2float(ah[i + j*ldah]));
	}
}

__global__
void zoomBack(int m,int n,float *dA,int lda,float mul)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
        dA[i + j*lda] *= mul;
	}
}

__global__
void zoomDevice(int m,int n,float *dA,int lda,float mul)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
        dA[i + j*lda] *= mul;
	}
}

__global__
void myslacpy( int m, int n, float *da, int lda, float *db, int ldb )
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i<m && j<n) {
		db[i+j*ldb] = da[i+j*lda];
	}
}

__global__
void myslacpy( int m, int n, double *da, int lda, double *db, int ldb )
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i<m && j<n) {
		db[i+j*ldb] = da[i+j*lda];
	}
}

void floatToHalf(int m, int n, float* dA, half* dhA)
{
    dim3 gridDim((m + 31) / 32, (n+ 31) / 32);
    dim3 blockDim(32, 32);
    s2h << <gridDim, blockDim >> > (m, n, dA, m, dhA, m);
}

void zoomHost(int m,int n,float *dA, int lda, float mul)
{
    dim3 gridDim((m + 31) / 32, (n+ 31) / 32);
    dim3 blockDim(32, 32);
    zoomDevice << <gridDim, blockDim >> > (m, n, dA, m,mul);
}

// clear 'L' lower/ 'U' Upper triangular part
__global__
void clear_tri(char uplo, int m, int n, float *a, int lda)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i<m && j<n) {
		if (uplo == 'l') {
			if (i>j) {
				a[i+j*lda] = 0;
			}
		} else {
			printf("clear_tri: option %c not implemented. \n", uplo);
			assert(0);
		}
	}
}

void generateMatrixDeviceUniform(float *dA,int m,int n)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    int seed = rand()%3000;
    printf("seed is %d\n",seed);
	curandSetPseudoRandomGeneratorSeed(gen, seed);
    //curandGenerateNormal(gen, dA, m*n,10,1000);
    curandGenerateUniform(gen,dA,m*n);

}

void generateMatrixDevice(float *dA,int m,int n)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    int seed = rand()%3000;
    printf("seed is %d\n",seed);
	curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, dA, m*n,0,1);
    //printMatrixDevice("t.csv",m,n,dA,m);
    //curandGenerateUniform(gen,dA,m*n);
}

void generateMatrixDevice(double *dA,int m,int n)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    int seed = rand()%3000;
    printf("seed is %d\n",seed);
    float *fA;
    cudaMalloc(&fA,sizeof(float)*m*n);
	curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, fA, m*n,0,1);
    dim3 grids2d( (m+31)/32, (n+31)/32 );
    dim3 blocks2d( 32, 32 );
    s2d<<<grids2d,blocks2d>>>( m, n, fA,m,dA,m);
    cudaFree(fA);
    //printMatrixDevice("t.csv",m,n,dA,m);
    //curandGenerateUniform(gen,dA,m*n);
}

void generateMatrixDevice(float *dA,int m,int n,int seed)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    printf("seed is %d\n",seed);
	curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, dA, m*n,0,1);
    printMatrixDevice("t.csv",m,n,dA,m);
    //curandGenerateUniform(gen,dA,m*n);
}
cudaEvent_t begin, end;
void startTimer()
{
    cudaEventCreate(&begin);
	cudaEventRecord(begin);
}

float stopTimer()
{
    cudaEventCreate(&end);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return milliseconds;
}

void startTimer(cudaEvent_t* start)
{
    cudaEventCreate(start);
	cudaEventRecord(*start);
}

float stopTimer(cudaEvent_t stop,cudaEvent_t start)
{
    cudaEventCreate(&stop);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
}

void host2Device(int m, int n, float *dA, float *hA)
{
    cudaMemcpy(dA,hA,sizeof(float)*m*n,cudaMemcpyHostToDevice);
}

void host2Device(int m, int n, double *dA, double *hA)
{
    cudaMemcpy(dA,hA,sizeof(double)*m*n,cudaMemcpyHostToDevice);
}

void clearTri(int m,int n, float* R, int ldr)
{
    dim3 grid( (m+1)/32, (n+1)/32 );
    dim3 block( 32, 32 );
    clear_tri<<<grid,block>>>('l', m, n, R, ldr);
    //printMatrixDevice("tmpR.csv",m,n,R,ldr);
}

void sgemm(int m,int n,int k,float *dA,int lda, float *dB,int ldb,float *dC, int ldc)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    float sone = 1;
    float szero = 0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
         m,n,k, 
         &sone, dA, lda, 
         dB, ldb, 
         &szero, dC, ldc);
}


void copyDevice(int m, int n, float* A, int lda, float* R, int ldr)
{
    dim3 grid( (m+31)/32, (n+31)/32 );
	dim3 block( 32, 32 );
	myslacpy<<<grid, block>>>(n, n, A, lda, R, ldr );
}

void copyDevice(int m, int n, double* A, int lda, double* R, int ldr)
{
    dim3 grid( (m+31)/32, (n+31)/32 );
	dim3 block( 32, 32 );
	myslacpy<<<grid, block>>>(n, n, A, lda, R, ldr );
}

void sgemm(int m,int n,int k,float *dA,int lda, float *dB,int ldb,float *dC, int ldc,float alpha,float beta)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    float sone = alpha;
    float szero = beta;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
         m,n,k, 
         &sone, dA, lda, 
         dB, ldb, 
         &szero, dC, ldc);
}

float snorm(int m,int n,float* dA)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    float sn;
    int incx = 1;
    cublasSnrm2(handle, m*n, dA, incx, &sn);
    return sn;
}


void snorm(int m,int n,float* dA, float* result)
{
    //printf("Perform double norm operation\n");
    cublasHandle_t handle;
    cublasCreate(&handle);
    int incx = 1;
    cublasSnrm2(handle, m*n, dA, incx, result);
    //printf("Ends double norm operation\n");
    
}

double dnorm(int m,int n,double* dA)
{
    //printf("Perform double norm operation\n");
    cublasHandle_t handle;
    cublasCreate(&handle);
    int incx = 1;
    double result;
    cublasDnrm2(handle, m*n, dA, incx, &result);
    //printf("Ends double norm operation\n");
    return result;
}

void dnorm(int m,int n,double* dA, double* result)
{
    //printf("Perform double norm operation\n");
    cublasHandle_t handle;
    cublasCreate(&handle);
    int incx = 1;
    cublasDnrm2(handle, m*n, dA, incx, result);
    //printf("Ends double norm operation\n");
}

void hnorm(int m, int n,__half* dA, float* result)
{
    float* dfA;
    cudaMalloc(&dfA,sizeof(float)*m*n);
    dim3 gridDim((m + 31) / 32, (n+ 31) / 32);
    dim3 blockDim(32, 32);
    h2s << <gridDim, blockDim >> > (m, n, dA, m,dfA,m);
    snorm(m,n,dfA,result);
}

void dsubstract(int m,int n, double* dA,int lda, double* dB, int ldb)
{
    //printf("Perform double substract operation\n");
    cublasHandle_t handle;
    cublasCreate(&handle);

    double dnegone = -1;
    double done = 1;
    cublasDgeam(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n,
        &done,
        dA, lda,
        &dnegone,
        dB, ldb,
        dA, lda);
}

void dsubstract(int m,int n, float* dA,int lda, float* dB, int ldb)
{
    //printf("Perform double substract operation\n");
    cublasHandle_t handle;
    cublasCreate(&handle);

    float snegone = -1;
    float sone = 1;
    cublasSgeam(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n,
        &sone,
        dA, lda,
        &snegone,
        dB, ldb,
        dA, lda);
}

void dgemm(int m,int n,int k,double *dA,int lda, double *dB,int ldb,double *dC, int ldc)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    double done = 1;
    double dzero = 0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
         m,n,k, 
         &done, dA, lda, 
         dB, ldb, 
         &dzero, dC, ldc);
}

void dgemm(int m,int n,int k,double *dA,int lda, double *dB,int ldb,double *dC, int ldc, double alpha, double beta)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    double done = alpha;
    double dzero = beta;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
         m,n,k, 
         &done, dA, lda, 
         dB, ldb, 
         &dzero, dC, ldc);
}

void tcgemm(int m,int n,int k, float* dA,int lda, float* dB, int ldb, float* dC,int ldc)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    __half *dhA,*dhB;
    cudaMalloc(&dhA,sizeof(__half)*m*k);
    cudaMalloc(&dhB,sizeof(__half)*k*n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, NULL);
    dim3 gridDim((m + 31) / 32, (k+ 31) / 32);
    dim3 blockDim(32, 32);
    s2h << <gridDim, blockDim >> > (m, k, dA, m, dhA, m);
    dim3 gridDimb((k + 31) / 32, (n+ 31) / 32);
    dim3 blockDimb(32, 32);
    s2h << <gridDimb, blockDimb >> > (k, n, dB, k, dhB, k);

    float sone = 1.0;
    float szero = 0.0;
   
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m,n,k,
        &sone, dhA, CUDA_R_16F, lda,
        dhB, CUDA_R_16F, ldb,
        &szero, dC, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("TensorCore takes %f ms\n",msecTotal);
    cudaFree(dhA);
    cudaFree(dhB);
}

void tcgemmRefine(int m,int n,int k, float* dA,int lda, float* dB, int ldb, float* dC,int ldc)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    __half *dhA,*dhB;
    cudaMalloc(&dhA,sizeof(__half)*m*k);
    cudaMalloc(&dhB,sizeof(__half)*k*n);

    __half *drA,*drB;
    cudaMalloc(&drA,sizeof(__half)*m*k);
    cudaMalloc(&drB,sizeof(__half)*k*n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, NULL);
    dim3 gridDim((m + 31) / 32, (k+ 31) / 32);
    dim3 blockDim(32, 32);
    s2hStoreResidule << <gridDim, blockDim >> > (m, k, dA, m, dhA, m,drA,m);
    dim3 gridDimb((k + 31) / 32, (n+ 31) / 32);
    dim3 blockDimb(32, 32);
    s2hStoreResidule << <gridDimb, blockDimb >> > (k, n, dB, k, dhB, k,drB,k);

    float res= 0;
    hnorm(m,k,drA,&res);
    printf("norm of residual A is %.6e\n",res/m/k);

    hnorm(k,n,drB,&res);
    printf("norm of residual B is %.6e\n",res/k/n);


    float sone = 1.0;
    float szero = 0.0;
   
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m,n,k,
        &sone, dhA, CUDA_R_16F, lda,
        dhB, CUDA_R_16F, ldb,
        &szero, dC, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m,n,k,
        &sone, drA, CUDA_R_16F, lda,
        dhB, CUDA_R_16F, ldb,
        &sone, dC, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);  
        
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m,n,k,
        &sone, dhA, CUDA_R_16F, lda,
        drB, CUDA_R_16F, ldb,
        &sone, dC, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT); 

    /*
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m,n,k,
        &sone, drA, CUDA_R_16F, lda,
        drB, CUDA_R_16F, ldb,
        &sone, dC, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);*/

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("TensorCore takes %f ms\n",msecTotal);
    cudaFree(dhA);
    cudaFree(dhB);
    cudaFree(drA);
    cudaFree(drB);
}

void tcgemmRefineFloat(int m,int n,int k, float* dA,int lda, float* dB, int ldb, float* dC,int ldc)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    __half *dhA,*dhB;
    cudaMalloc(&dhA,sizeof(__half)*m*k);
    cudaMalloc(&dhB,sizeof(__half)*k*n);

    float *drA,*drB;
    cudaMalloc(&drA,sizeof(float)*m*k);
    cudaMalloc(&drB,sizeof(float)*k*n);

    float *dfA,*dfB;
    cudaMalloc(&dfA,sizeof(float)*m*k);
    cudaMalloc(&dfB,sizeof(float)*k*n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, NULL);
    dim3 gridDim((m + 31) / 32, (k+ 31) / 32);
    dim3 blockDim(32, 32);
    s2hStoreResidule << <gridDim, blockDim >> > (m, k, dA, m, dhA, m,drA,m);
    h2s<< <gridDim, blockDim >> > (m, k, dhA, m, dfA, m);
    dim3 gridDimb((k + 31) / 32, (n+ 31) / 32);
    dim3 blockDimb(32, 32);
    s2hStoreResidule << <gridDimb, blockDimb >> > (k, n, dB, k, dhB, k,drB,k);
    h2s<< <gridDimb, blockDimb >> > (k, n, dhB, k, dfB, k);

    float res= 0;
    snorm(m,k,drA,&res);
    printf("norm of residual A is %.6e\n",res/m/k);

    snorm(k,n,drB,&res);
    printf("norm of residual B is %.6e\n",res/k/n);


    float sone = 1.0;
    float szero = 0.0;

    //sgemm(m,n,k,dA,m,dB,k,dC,m);
    //sgemm(m,n,k,dfA,m,dfB,k,dC,m);

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m,n,k,
        &sone, dhA, CUDA_R_16F, lda,
        dhB, CUDA_R_16F, ldb,
        &szero, dC, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    /*
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m,n,k,
        &sone, dhA, CUDA_R_16F, lda,
        dhB, CUDA_R_16F, ldb,
        &szero, dC, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);*/
    
        //sgemm(m,n,k,dfA,m,drB,k,dC,m,1.0,1.0);
        //sgemm(m,n,k,drA,m,dfB,k,dC,m,1.0,1.0);
    
    /*
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m,n,k,
        &sone, drA, CUDA_R_32F, lda,
        dfB, CUDA_R_32F, ldb,
        &sone, dC, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);  
        
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m,n,k,
        &sone, dfA, CUDA_R_32F, lda,
        drB, CUDA_R_32F, ldb,
        &sone, dC, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);*/

    /*
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m,n,k,
        &sone, drA, CUDA_R_32F, lda,
        drB, CUDA_R_32F, ldb,
        &sone, dC, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);*/
    
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("TensorCore takes %f ms\n",msecTotal);
    cudaFree(dhA);
    cudaFree(dhB);
    cudaFree(drA);
    cudaFree(drB);
    cudaFree(dfA);
    cudaFree(dfB);
}

void tcgemmRefineZoom(int m,int n,int k, float* dA,int lda, float* dB, int ldb, float* dC,int ldc,float mul)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    __half *dhA,*dhB;
    cudaMalloc(&dhA,sizeof(__half)*m*k);
    cudaMalloc(&dhB,sizeof(__half)*k*n);

    float *drA,*drB;
    cudaMalloc(&drA,sizeof(float)*m*k);
    cudaMalloc(&drB,sizeof(float)*k*n);

    float *dfA,*dfB;
    cudaMalloc(&dfA,sizeof(float)*m*k);
    cudaMalloc(&dfB,sizeof(float)*k*n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, NULL);
    dim3 gridDim((m + 31) / 32, (k+ 31) / 32);
    dim3 blockDim(32, 32);
    s2hStoreResidule << <gridDim, blockDim >> > (m, k, dA, m, dhA, m,drA,m);
    h2s<< <gridDim, blockDim >> > (m, k, dhA, m, dfA, m);
    dim3 gridDimb((k + 31) / 32, (n+ 31) / 32);
    dim3 blockDimb(32, 32);
    s2hStoreResidule << <gridDimb, blockDimb >> > (k, n, dB, k, dhB, k,drB,k);
    h2s<< <gridDimb, blockDimb >> > (k, n, dhB, k, dfB, k);

    float res= 0;
    snorm(m,k,drA,&res);
    printf("norm of residual A is %.6e\n",res/m/k);

    snorm(k,n,drB,&res);
    printf("norm of residual B is %.6e\n",res/k/n);


    float sone = 1.0;
    float szero = 0.0;

    //sgemm(m,n,k,dA,m,dB,k,dC,m);
    //sgemm(m,n,k,dfA,m,dfB,k,dC,m);

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m,n,k,
        &sone, dA, CUDA_R_32F, lda,
        dB, CUDA_R_32F, ldb,
        &szero, dC, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    /*
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m,n,k,
        &sone, dhA, CUDA_R_16F, lda,
        dhB, CUDA_R_16F, ldb,
        &szero, dC, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);*/
    
        //sgemm(m,n,k,dfA,m,drB,k,dC,m,1.0,1.0);
        //sgemm(m,n,k,drA,m,dfB,k,dC,m,1.0,1.0);
    
    /*
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m,n,k,
        &sone, drA, CUDA_R_32F, lda,
        dfB, CUDA_R_32F, ldb,
        &sone, dC, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);  
        
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m,n,k,
        &sone, dfA, CUDA_R_32F, lda,
        drB, CUDA_R_32F, ldb,
        &sone, dC, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);*/

    /*
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m,n,k,
        &sone, drA, CUDA_R_32F, lda,
        drB, CUDA_R_32F, ldb,
        &sone, dC, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);*/
    
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("TensorCore takes %f ms\n",msecTotal);
    cudaFree(dhA);
    cudaFree(dhB);
    cudaFree(drA);
    cudaFree(drB);
    cudaFree(dfA);
    cudaFree(dfB);
}


void getDoubleResult(int m,int n, int k,float *dA,int lda, float *dB, int ldb, double *ddC, int ldc)
{
    double *ddA,*ddB;
    cudaMalloc(&ddA,sizeof(double)*m*k);
    cudaMalloc(&ddB,sizeof(double)*k*n);

    dim3 gridDim((m + 31) / 32, (k + 31) / 32);
    dim3 blockDim(32, 32);

    s2d << <gridDim, blockDim >> > (m, k, dA, m, ddA, m);

    dim3 gridDimb((k + 31) / 32, (n + 31) / 32);
    dim3 blockDimb(32, 32);
    s2d << <gridDimb, blockDimb >> > (k, n, dB, k, ddB, k);
    
    dgemm(m,n,k,ddA,m,ddB,k,ddC,m);
    cudaFree(ddA);
    cudaFree(ddB);    
}

double checkResult(int m,int n, float*dC, double *ddC, int ldc)
{
    double *dfC;
    cudaMalloc(&dfC,sizeof(double)*m*n);
    dim3 gridDim((m + 31) / 32, (n+ 31) / 32);
    dim3 blockDim(32, 32);
    s2d << <gridDim, blockDim >> > (m, n, dC, m, dfC, m);

    dsubstract(m,n,dfC,ldc,ddC,ldc);
    double result;
    dnorm(m,n,dfC,&result);
    cudaFree(dfC);
    return result;
}
