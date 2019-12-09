#include <iostream>

__global__
void seteye( int m, int n, float *a, int lda)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		if (i == j) 
			a[i+j*lda] = 1;
		else
			a[i+j*lda] = 0;
	}

}

void qrSolve(int m, int n, float *A, int lda, float *R, int ldr, float *reflector, float *d_tau,float *d_work, int lwork)
{
    printf("Function qrSolve\n");
    cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    

	int *devInfo = 0;
    cudaMalloc ((void**)&devInfo, sizeof(int));
    cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    
    //cout << "lwork/4 (elms)=" << lwork/4 << endl;
    
    //cudaEvent_t start, stop;
    startTimer();

    cusolver_status = cusolverDnSgeqrf(
        cusolverH, 
        m, 
        n, 
        A, 
        lda, 
        d_tau, 
        d_work, 
        lwork, 
        devInfo);

    printf("Sgeqrf return code %d\n", cusolver_status);

    float milliseconds =  stopTimer();

    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    printf("cuSolver: sgeqrf (%d,%d) takes %f ms\n",m,n,milliseconds);
    
    printf("TFLOPS/s: %f\n",(2.0*m*n*n - 2.0/3*n*n*n)/milliseconds*1000/1e12);

    printf("INFO %d\n",info);

    copyDevice(m,n,A,lda,R,ldr);
    clearTri(n,n,R,ldr);
    //printMatrixDevice("RR.csv",n,n,R,n);

    float *tA;
    cudaMalloc(&tA,sizeof(float)*m*n);
    cudaMemcpy(tA,A,sizeof(float)*m*n,cudaMemcpyDeviceToDevice);
    cudaMemcpy(reflector,A,sizeof(float)*m*n,cudaMemcpyDeviceToDevice);
    dim3 grid222( (m+1)/32, (n+1)/32 );
    dim3 block222( 32, 32 );
    seteye<<<grid222,block222>>>( m, n, tA, m);
    //printMatrixDevice("tA.csv",m,n,tA,m);

    startTimer();
    cusolver_status= cusolverDnSormqr(
        cusolverH,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_N,
        m,
        n,
        n,
        A,
        lda,
        d_tau,
        tA,
        m,
        d_work,
        lwork,
        devInfo);
    milliseconds = stopTimer();
    printf("Obtaining Q takes %f ms\n",milliseconds);
    cudaMemcpy(A,tA,sizeof(float)*m*n,cudaMemcpyDeviceToDevice);
    cudaFree(tA);
}

void sllsSolve(int m,int n,float* reflector, int lda, float* dtau,float* d_work,int lwork, float *R, float *dfx)
{
    printf("1\n");
    int *devInfo = 0;
    cudaMalloc ((void**)&devInfo, sizeof(int));

    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnCreate(&cusolverH);

    cublasHandle_t handle;
    cublasCreate(&handle);
    

    float hfb[m];
    for(int i=0;i<m;i++)
    {
        hfb[i] = 1.0;
    }
    float *dfb;
    cudaMalloc(&dfb,sizeof(float)*m);
    
    cudaMemcpy(dfb,hfb,sizeof(float)*m,cudaMemcpyHostToDevice);
    //free(hfb);
    printf("2\n");

    startTimer();
    cusolverDnSormqr(
        cusolverH,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,
        m,
        1,
        n,
        reflector,
        lda,
        dtau,
        dfb,
        m,
        d_work,
        lwork,
        devInfo);

    dim3 grids2d2( (n+31)/32, (1+31)/32 );
    dim3 blocks2d2( 32, 32 );
    myslacpy<<<grids2d2,blocks2d2>>>(n, 1, dfb, m, dfx, n );
    float sone = 1.0;
    cublasStrsm(handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,n,1,&sone,R,n,dfx,n);
    float milliseconds;
    milliseconds = stopTimer();
    printf("single LLS direct solver takes %fms\n",milliseconds);
    cudaFree(dfb);
}

void checkResult(int m, int n, float *A, int lda,float *x)
{
    startTimer();
    double *db,*dx;
    double hb[m];
    for(int i=0;i<m;i++)
    {
        hb[i] = 1.0;
    }
    cublasHandle_t handle;
    cublasCreate(&handle);
    double normx;
    cudaMalloc(&db,sizeof(double)*m);
    cudaMalloc(&dx,sizeof(double)*n);
    cudaMemcpy(db,hb,m*sizeof(double),cudaMemcpyHostToDevice);

    double done = 1.0;
    double dnegone = -1.0;
    double dzero = 0.0;
    
    double *doubleA;
    cudaMalloc(&doubleA,sizeof(double)*m*n);

    dim3 grids2d( (m+31)/32, (n+31)/32 );
    dim3 blocks2d( 32, 32 );
    s2d<<<grids2d,blocks2d>>>( m, n, A,m,doubleA,m);
    cudaFree(A);

    double *doubleX;
    cudaMalloc(&doubleX,sizeof(double)*n);
    float milliseconds  = stopTimer();
    printf("Phase 1 takes %fms\n",milliseconds);
    dim3 grids2d2( (n+31)/32, (1+31)/32 );
    dim3 blocks2d2( 32, 32 );
    s2d<<<grids2d2,blocks2d2>>>( n, 1, x,n,doubleX,n);
    cudaFree(x);

    startTimer();
    double *tempx;
    cudaMalloc(&tempx,sizeof(double)*n);
    cublasDgemv(handle,CUBLAS_OP_T,m,n,&done,doubleA,m,db,1,&dzero,tempx,1);
    double normb;
    normb = dnorm(m,1,tempx);
    cudaFree(tempx);

    cublasDgemv(handle,CUBLAS_OP_N,m,n,&done,doubleA,m,doubleX,1,&dnegone,db,1);
    cublasDgemv(handle,CUBLAS_OP_T,m,n,&done,doubleA,m,db,1,&dzero,doubleX,1);
    double normsol = dnorm(n,1,doubleX);
    printf("||A^T*(Ax-b)||/(||A^T*b||) = %.6e\n",normsol/normb);
    milliseconds = stopTimer();
    printf("Phase 2 takes %fms\n",milliseconds);
    
}


void sSolveLLS(int m,int n, float* hA,int lda)
{
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnCreate(&cusolverH);

    printf("Function sSolveLLS\n");
    float *dA;
    cudaMalloc(&dA,sizeof(float)*n*m);
    host2Device(m,n,dA,hA);
    float *R;
    cudaMalloc(&R,sizeof(float)*n*n);

    float *reflector;
    cudaMalloc(&reflector,sizeof(float)*m*n);

    float *d_tau;
    cudaMalloc(&d_tau, sizeof(float)*n);
    
    float *d_work;
    int lwork = 0;
    
    cusolverDnSgeqrf_bufferSize(
		cusolverH, 
		m, 
		n, 
		dA, 
		lda, 
        &lwork);
    cudaMalloc((void**)&d_work, sizeof(float)*lwork);

    qrSolve(m,n,dA,m,R,n,reflector,d_tau,d_work,lwork);
    //printMatrixDevice("Q.csv",m,n,dA,m);
    //printMatrixDevice("R.csv",n,n,R,n);
    float *A;
    cudaMalloc(&A,sizeof(float)*n*m);
    host2Device(m,n,A,hA);
    //printMatrixDevice("A.csv",m,n,A,m);

    float *dfx;
    cudaMalloc(&dfx,sizeof(float)*n);

    //check QR result
    float normA = snorm(m,n,A);
    sgemm(m,n,n,dA,m,R,n,A,m,1.0,-1.0);
    float normResult = snorm(m,n,A);
    printf("||A-QR||/||A|| is %.3e\n", normResult/normA);
    cudaFree(dA);

    sllsSolve(m,n,reflector, lda, d_tau, d_work, lwork, R, dfx);
    host2Device(m,n,A,hA);
    //printMatrixDevice("A.csv",m, n, A, m);
    //printMatrixDevice("x.csv",n, 1, dfx, n);
    checkResult(m, n, A, lda,dfx);
   
}