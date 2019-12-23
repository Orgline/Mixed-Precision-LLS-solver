#include <iostream>
#include "cub-1.8.0/cub-1.8.0/cub/cub.cuh"

#define NMIN 128

struct cudaCtxt {
	cublasHandle_t cublas_handle;
	cusolverDnHandle_t cusolver_handle;
};

struct F4add
{
    __host__ __device__ __forceinline__
    float4 operator()(const float4& a, const float4& b) const 
    {
    // return a*a;
    float4 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;
    return c;
    }
};
// Each TB factorizes a 256*n submatrix; AA will be overwritten with Q, RR overwritten with R.
// n <= 32. m is supposed to be any number.
__global__ void geqrf_tb_256x32_multicub(int m, int n, float *AA, int lda, float *RR, int ldr)
{
    if (n>32) 
    {
        if (threadIdx.x+blockDim.x*blockIdx.x == 0)
            printf("geqrf_tb_256x32: only n<=32 supported. current n=%d\n. Returning.", n);
        return;
    }
    int mm = m - blockIdx.x*256; // TB local number of rows
    mm = (mm<256) ? mm : 256;

    const int mnmin = (mm<n) ? mm : n;

    float *A = &AA[blockIdx.x*256];
    float *R = &RR[blockIdx.x*32];
    //load from global memory to shared memory
    __shared__ float As[256*32], Rs[32*32];
    const int i = threadIdx.x;

    #pragma unroll
    for (int j=0; j<n; j++) 
    {
        if (i<mm) As[i+j*256] = A[i+j*lda];
    }
    __syncthreads();

    const int ldas = 256, ldrs = 32;


    float acc1[1], acc2[1], acc3[1], acc4[1];
    float sum1, sum2, sum3, sum4;

    typedef cub::BlockReduce<float4, 256> BlockReduce;
    typedef cub::BlockReduce<float, 256> BlockReduce2;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ typename BlockReduce2::TempStorage temp_storage2;

    for (int k=0; k<mnmin; k++)
    {
        sum1 = (i<mm) ? As[i+k*ldas]*As[i+k*ldas] : 0;
        float sumsqr = BlockReduce2(temp_storage2).Sum(sum1);

        if (i==0) 
        {
            Rs[k+k*ldrs] = sqrt( sumsqr );
        }
        __syncthreads();

        if (i<mm) 
            As[i+k*ldas] = As[i+k*ldas] / Rs[k+k*ldrs];

        for (int j=k+1; j<(k+4)/4*4 && j<n; j++) 
        {
            sum1 = (i<mm) ? (As[i+k*ldas] * As[i+j*ldas]) : 0;
            float sum = BlockReduce2(temp_storage2).Sum(sum1);
            if (i==0)
                Rs[k+j*ldrs] = sum;
        }

        for (int j=(k+4)/4*4; j<n; j+=4) 
        {
            float4 S;
            S.x = (i<mm) ? (As[i+k*ldas] * As[i+j*ldas]) :    0 ;
            S.y = (i<mm) ? (As[i+k*ldas] * As[i+(j+1)*ldas]): 0 ;
            S.z = (i<mm) ? (As[i+k*ldas] * As[i+(j+2)*ldas]): 0 ;
            S.w = (i<mm) ? (As[i+k*ldas] * As[i+(j+3)*ldas]): 0 ;
            S = BlockReduce(temp_storage).Reduce(S, F4add());
            if (i==0) 
            {
                Rs[k+j*ldrs] = S.x;
                Rs[k+(j+1)*ldrs] = S.y;
                Rs[k+(j+2)*ldrs] = S.z;
                Rs[k+(j+3)*ldrs] = S.w;
            }
        }    

        __syncthreads();

        #pragma unroll
        for (int j=k+1; j<n; j++)
            if (i<mm) 
                As[i+j*ldas] -= As[i+k*ldas]*Rs[k+j*ldrs];

    }
    #pragma unroll
    for (int j=0; j<n; j++) 
    {
        if( i<mm) 
            A[i+j*lda] = As[i+j*ldas];
        if (i<=j) 
        {
            if (i<mm && i<n) 
                R[i+j*ldr] = Rs[i+j*ldrs];
        } 
        else 
        {
            if (i<mm && i<n) 
                R[i+j*ldr] = 0;
        }
    }
}

void CAQR_256x32(cudaCtxt ctxt, int m, int n, float *A, int lda, float *R, int ldr, float *work)
{
    //printf("Function CAQR_256x32\n");

    if (n!=32) 
    {
        printf("[Error]: CAQR_32 does not support n!=32\n");
        assert(0);
        return;
    }


    if (m <= 256) 
    {
        // printf("CAQR: Recursion tree leaf: ");
        geqrf_tb_256x32_multicub<<<1,256>>>(m, n, A,  lda, R, ldr);
    } 
    else 
    { // m > 256, recurse.
        if (m%256 != 0) 
        {
            printf("[Error]: CAQR_32 m must be multiple of 256. m=%d \n", m);
            assert(0);
        }
        int ldwork = m/256*32;
        int mm = m/256*32;

        geqrf_tb_256x32_multicub<<<m/256,256>>>(m, n, A,  lda, work, ldwork);

        CAQR_256x32(ctxt, mm , n, work, ldwork, R, ldr, work+ldwork*n);

        float sone = 1.0, szero = 0.0;
        cublasSgemmStridedBatched(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        256, 32, 32,
        &sone, A, lda, 256,
        work, ldwork, 32,
        &szero, work+ldwork*32,lda, 256,
        m/256);
        dim3 grid( (m+31)/32, (n+31)/32 );
        dim3 block( 32, 32 );
        myslacpy<<<grid, block>>>( m, n, work+ldwork*32, lda, A,  lda );
    }

}

void CAQR_256x128_TC(cudaCtxt ctxt, int m, int n, float *A, int lda, float *R, int ldr, float *work)
{
    //printf("Function CAQR_256x128_TC\n");
    //printf("CAQR128: m, n, lda, ldr = %d, %d, %d, %d\n", m, n, lda, ldr);
    if (m<256 || n!=128) 
    {
        printf("CAQR_256x128: ERROR: m must be > 256, n must be 128. (m,n)=(%d,%d)\n", m, n);
    }
    float sone = 1.0, szero = 0.0, snegone = -1.0;
// QR left half 64
    CAQR_256x32(ctxt, m, 32, A, lda, R, ldr, work);
    /*
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
    32, 32, m,
    &sone, A, lda,
    &A[32*lda], lda,
    &szero, &R[32*ldr], ldr);
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
    m, 32, 32,
    &snegone, A, lda,
    &R[32*ldr], ldr,
    &sone, &A[32*lda], lda);*/
    cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        32, 32, m,
        &sone, A, CUDA_R_16F,lda,
        &A[32*lda], CUDA_R_16F,lda,
        &szero, &R[32*ldr], CUDA_R_32F,ldr,
        CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, 32, 32,
        &snegone, A, CUDA_R_16F,lda,
        &R[32*ldr], CUDA_R_16F,ldr,
        &sone, &A[32*lda], CUDA_R_32F,lda,
        CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    CAQR_256x32(ctxt, m, 32, &A[32*lda], lda, &R[32+32*ldr], ldr, work);
// update trailing 64
    /*
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
    64, 64, m,
    &sone, A, lda,
    &A[64*lda], lda,
    &szero, &R[64*ldr], ldr);
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
    m, 64, 64,
    &snegone, A, lda,
    &R[64*ldr], ldr,
    &sone, &A[64*lda], lda);*/
    cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        64, 64, m,
        &sone, A, CUDA_R_32F,lda,
        &A[64*lda], CUDA_R_16F,lda,
        &szero, &R[64*ldr], CUDA_R_16F,ldr,
        CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, 64, 64,
        &snegone, A, CUDA_R_16F,lda,
        &R[64*ldr], CUDA_R_16F,ldr,
        &sone, &A[64*lda], CUDA_R_32F,lda,
        CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);

// QR right half 64
    A = &A[64*lda]; R = &R[64*ldr+64];
    CAQR_256x32(ctxt, m, 32, A, lda, R, ldr, work);
    /*
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
    32, 32, m,
    &sone, A, lda,
    &A[32*lda], lda,
    &szero, &R[32*ldr], ldr);
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
    m, 32, 32,
    &snegone, A, lda,
    &R[32*ldr], ldr,
    &sone, &A[32*lda], lda);*/
    cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        32, 32, m,
        &sone, A, CUDA_R_16F,lda,
        &A[32*lda], CUDA_R_16F,lda,
        &szero, &R[32*ldr], CUDA_R_32F,ldr,
        CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, 32, 32,
        &snegone, A, CUDA_R_16F,lda,
        &R[32*ldr], CUDA_R_16F,ldr,
        &sone, &A[32*lda], CUDA_R_32F,lda,
        CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CAQR_256x32(ctxt, m, 32, &A[32*lda], lda, &R[32+32*ldr], ldr, work);
}

void CAQR_256x128(cudaCtxt ctxt, int m, int n, float *A, int lda, float *R, int ldr, float *work)
{
    //printf("Function CAQR_256x128\n");
    //printf("CAQR128: m, n, lda, ldr = %d, %d, %d, %d\n", m, n, lda, ldr);
    if (m<256 || n!=128) 
    {
        printf("CAQR_256x128: ERROR: m must be > 256, n must be 128. (m,n)=(%d,%d)\n", m, n);
    }
    float sone = 1.0, szero = 0.0, snegone = -1.0;
// QR left half 64
    CAQR_256x32(ctxt, m, 32, A, lda, R, ldr, work);
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
    32, 32, m,
    &sone, A, lda,
    &A[32*lda], lda,
    &szero, &R[32*ldr], ldr);
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
    m, 32, 32,
    &snegone, A, lda,
    &R[32*ldr], ldr,
    &sone, &A[32*lda], lda);
    CAQR_256x32(ctxt, m, 32, &A[32*lda], lda, &R[32+32*ldr], ldr, work);
// update trailing 64
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
    64, 64, m,
    &sone, A, lda,
    &A[64*lda], lda,
    &szero, &R[64*ldr], ldr);
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
    m, 64, 64,
    &snegone, A, lda,
    &R[64*ldr], ldr,
    &sone, &A[64*lda], lda);
// QR right half 64
    A = &A[64*lda]; R = &R[64*ldr+64];
    CAQR_256x32(ctxt, m, 32, A, lda, R, ldr, work);
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
    32, 32, m,
    &sone, A, lda,
    &A[32*lda], lda,
    &szero, &R[32*ldr], ldr);
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
    m, 32, 32,
    &snegone, A, lda,
    &R[32*ldr], ldr,
    &sone, &A[32*lda], lda);
    CAQR_256x32(ctxt, m, 32, &A[32*lda], lda, &R[32+32*ldr], ldr, work);
}

// recursive QR V2: use the CAQR as panel
// QR(A): A will be overwritten with Q, and R will be populated.
// A (in,out): m*n, on device
// R (out): n*n, on device (provided by caller). Upon return, its upper triangular
//               contains the R factor.
// lda: >=m
// ldr: >=n
// work: <--m*NMIN--> <-NMIN-> <--lwork_geqrf-->
//         ormqr,B       tau        geqrf, ormqr
// hwork: m*n in half precision
int RGEQRF2( cudaCtxt ctxt, int m, int n, float *A, int lda, float *R, int ldr, float *work, int lwork,
    __half *hwork, int lhwork)
{
    //printf("Function RGEQRF2\n");
    int info;
    
    // base case for recursion;
    if (n == 128) 
    {
        // GEQRF8( ctxt, m, n, A, lda, R, ldr, work, lwork );
        CAQR_256x128( ctxt, m, n, A, lda, R, ldr, work );
        return 0;
    }
    
    // left recurse
    RGEQRF2( ctxt, m, n/2, A, lda, R, ldr, work, lwork, hwork, lhwork );
    
    // ==========update trailing matrix ==========
    
    float sone = 1.0, szero = 0;
    // CUBLAS_CALL(cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n/2, n/2, m,
    //                          &sone, A, CUDA_R_32F, lda, &A[n/2*lda], CUDA_R_32F, lda,
    //                          &szero, &R[n/2*ldr], CUDA_R_32F, ldr, CUDA_R_32F,
    //                          // CUBLAS_GEMM_DEFAULT));
    //                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    // __half *Ah, *Bh, *Ch;
    // cudaMalloc( &Ah, sizeof(float)*m*n );
    // cudaMalloc( &Bh, sizeof(float)*m*n );
    __half *Ah = hwork;
    __half *Bh = &hwork[m*n/2];
    dim3 gridDim((m+31)/32,(n+31)/32);
    dim3 blockDim(32,32);
    s2h<<<gridDim, blockDim>>>(m, n/2, A, m, Ah, m);
    s2h<<<gridDim, blockDim>>>(m, n/2, &A[n/2*lda], m, Bh, m);
    cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n/2, n/2, m,
    &sone, Ah, CUDA_R_16F, lda, Bh, CUDA_R_16F, lda,
    &szero, &R[n/2*ldr], CUDA_R_32F, ldr, CUDA_R_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    float snegone= -1.0;
    dim3 gridDim2( (n+31)/32, (n+31)/31 );
    s2h<<<gridDim2, blockDim>>>(n/2, n/2, &R[n/2*ldr], ldr, Bh, n/2);
    cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n/2, n/2,
    &snegone, Ah, CUDA_R_16F, m, Bh, CUDA_R_16F, n/2,
    &sone, &A[n/2*lda], CUDA_R_32F, lda, CUDA_R_32F,
    // CUBLAS_GEMM_DEFAULT));
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    // right recurse
    RGEQRF2( ctxt, m, n/2, &A[n/2*lda], lda, &R[n/2+n/2*ldr], ldr, work, lwork, hwork, lhwork );
    return info;
}

void checkOtho(int m,int n,float *Q, int ldq)
{
    float *I;
    cudaMalloc(&I,sizeof(float)*n*n);
      
	dim3 grid96( (n+1)/32, (n+1)/32 );
	dim3 block96( 32, 32 );
    seteye<<<grid96,block96>>>( n, n, I, n);
    float snegone = -1.0;
    float sone  = 1.0;
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
        &snegone, Q, CUDA_R_32F, ldq, Q, CUDA_R_32F, ldq,
        &sone, I, CUDA_R_32F, n, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT);
    
    float normRes = snorm(n,n,I);
    printf("||I-Q'*Q||/N = %.6e\n",normRes/n);
    cudaFree(I);
}

void checkResult(int m,int n,float* A,int lda, float *Q, int ldq, float *R, int ldr)
{
    float normA = snorm(m,n,A);
    float alpha = 1.0;
    float beta = -1.0;
    sgemm(m,n,n,Q,ldq,R,ldr,A,lda,alpha,beta);
    float normRes = snorm(m,n,A);
    printf("||A-QR||/(||A||) = %.6e\n",normRes/normA);
}
    

void recursiveQR(int m,int n,float *hA,int lda)
{
    printf("Function recursive QR\n");
    cudaCtxt ctxt;
    cublasCreate( & ctxt.cublas_handle );
    cusolverDnCreate( & ctxt.cusolver_handle );
    int lwork;
    float* A;
    int ldr = n;
    cudaMalloc(&A,sizeof(float)*m*n);
    cudaMemcpy( A, hA, sizeof(float)*m*n, cudaMemcpyHostToDevice );
    printMatrixDevice("A.csv",m,n,A,m);
    
    cusolverDnSgeqrf_bufferSize(
        ctxt.cusolver_handle,
        m,
        NMIN,
        A,
        lda,
        &lwork);
    lwork += NMIN + m*NMIN;

    

    float *work;
    cudaMalloc( &work, lwork * sizeof(float) );

    
    __half *hwork;
	int lhwork = m*n;
    cudaMalloc( &hwork, sizeof(__half) * lhwork );

    float *R;
    cudaMalloc(&R,sizeof(float)*n*n);
    
    startTimer();
    RGEQRF2( ctxt, m, n, A, m, R, ldr, work, lwork, hwork, lhwork );
    float milliseconds = stopTimer();
    printf("RGEQRF takes %fms\n",milliseconds);

    float *Q;
    cudaMalloc(&Q,sizeof(float)*m*n);
    cudaMemcpy(Q,A,sizeof(float)*m*n,cudaMemcpyDeviceToDevice);
    cudaMemcpy(A, hA, sizeof(float)*m*n, cudaMemcpyHostToDevice);

    checkResult(m,n,A,lda,Q,m,R,n);

    checkOtho(m,n,Q,m);
    //printMatrixDevice("Q.csv",m,n,A,m);
    printMatrixDevice("R.csv",n,n,R,n);
}
