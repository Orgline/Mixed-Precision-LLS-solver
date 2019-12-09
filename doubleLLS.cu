#include <iostream>

__global__
void seteye( int m, int n, double *a, int lda)
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

__global__
void clear_tri(char uplo, int m, int n, double *a, int lda)
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

void clearTri(int m,int n, double* R, int ldr)
{
    dim3 grid( (m+1)/32, (n+1)/32 );
    dim3 block( 32, 32 );
    clear_tri<<<grid,block>>>('l', m, n, R, ldr);
    //printMatrixDevice("tmpR.csv",m,n,R,ldr);
}



void qrSolve(int m, int n, double *A, int lda, double *R, int ldr, double *reflector, double *d_tau,double *d_work, int lwork)
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

    cusolver_status = cusolverDnDgeqrf(
        cusolverH, 
        m, 
        n, 
        A, 
        lda, 
        d_tau, 
        d_work, 
        lwork, 
        devInfo);

    printf("Dgeqrf return code %d\n", cusolver_status);

    float milliseconds =  stopTimer();

    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    printf("cuSolver: dgeqrf (%d,%d) takes %f ms\n",m,n,milliseconds);
    
    printf("TFLOPS/s: %f\n",(2.0*m*n*n - 2.0/3*n*n*n)/milliseconds*1000/1e12);

    printf("INFO %d\n",info);

    copyDevice(m,n,A,lda,R,ldr);
    clearTri(n,n,R,ldr);
    //printMatrixDevice("RR.csv",n,n,R,n);

    double *tA;
    cudaMalloc(&tA,sizeof(double)*m*n);
    cudaMemcpy(tA,A,sizeof(double)*m*n,cudaMemcpyDeviceToDevice);
    cudaMemcpy(reflector,A,sizeof(double)*m*n,cudaMemcpyDeviceToDevice);
    dim3 grid222( (m+1)/32, (n+1)/32 );
    dim3 block222( 32, 32 );
    seteye<<<grid222,block222>>>( m, n, tA, m);
    //printMatrixDevice("tA.csv",m,n,tA,m);

    startTimer();
    cusolver_status= cusolverDnDormqr(
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
    cudaMemcpy(A,tA,sizeof(double)*m*n,cudaMemcpyDeviceToDevice);
    cudaFree(tA);
}

void sllsSolve(int m,int n,double* reflector, int lda, double* dtau,double* d_work,int lwork, double *R, double *dfx)
{
    printf("1\n");
    int *devInfo = 0;
    cudaMalloc ((void**)&devInfo, sizeof(int));

    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnCreate(&cusolverH);

    cublasHandle_t handle;
    cublasCreate(&handle);
    

    double hfb[m];
    for(int i=0;i<m;i++)
    {
        hfb[i] = 1.0;
    }
    double *dfb;
    cudaMalloc(&dfb,sizeof(double)*m);
    
    cudaMemcpy(dfb,hfb,sizeof(double)*m,cudaMemcpyHostToDevice);
    //free(hfb);
    printf("2\n");

    startTimer();
    cusolverDnDormqr(
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

    double sone = 1.0;
    cublasDtrsm(handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,n,1,&sone,R,n,dfx,n);
    float milliseconds;
    milliseconds = stopTimer();
    printf("single LLS direct solver takes %fms\n",milliseconds);
    cudaFree(dfb);
}

void checkResult(int m, int n, double *A, int lda,double *x)
{
    
    double *db,*dx;
    double hb[m];
    for(int i=0;i<m;i++)
    {
        hb[i] = 1.0;
    }
    cublasHandle_t handle;
    cublasCreate(&handle);
    //double normx;
    cudaMalloc(&db,sizeof(double)*m);
    cudaMalloc(&dx,sizeof(double)*n);
    cudaMemcpy(db,hb,m*sizeof(double),cudaMemcpyHostToDevice);

    double done = 1.0;
    double dnegone = -1.0;
    double dzero = 0.0;
    

    double *tempx;
    cudaMalloc(&tempx,sizeof(double)*n);
    cublasDgemv(handle,CUBLAS_OP_T,m,n,&done,A,m,db,1,&dzero,tempx,1);
    double normb;
    normb = dnorm(m,1,tempx);
    cudaFree(tempx);

    cublasDgemv(handle,CUBLAS_OP_N,m,n,&done,A,m,x,1,&dnegone,db,1);
    cublasDgemv(handle,CUBLAS_OP_T,m,n,&done,A,m,db,1,&dzero,x,1);
    double normsol = dnorm(n,1,x);
    printf("||A^T*(Ax-b)||/(||A^T*b||) = %.6e\n",normsol/normb);
    
}


void dSolveLLS(int m,int n, double* hA,int lda)
{
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnCreate(&cusolverH);

    printf("Function sSolveLLS\n");
    double *dA;
    cudaMalloc(&dA,sizeof(double)*n*m);
    host2Device(m,n,dA,hA);
    double *R;
    cudaMalloc(&R,sizeof(double)*n*n);

    double *reflector;
    cudaMalloc(&reflector,sizeof(double)*m*n);

    double *d_tau;
    cudaMalloc(&d_tau, sizeof(double)*n);
    
    double*d_work;
    int lwork = 0;
    
    cusolverDnDgeqrf_bufferSize(
		cusolverH, 
		m, 
		n, 
		dA, 
		lda, 
        &lwork);
    cudaMalloc((void**)&d_work, sizeof(double)*lwork);

    qrSolve(m,n,dA,m,R,n,reflector,d_tau,d_work,lwork);
    //printMatrixDevice("Q.csv",m,n,dA,m);
    //printMatrixDevice("R.csv",n,n,R,n);
    double *A;
    cudaMalloc(&A,sizeof(double)*n*m);
    host2Device(m,n,A,hA);
    //printMatrixDevice("A.csv",m,n,A,m);

    double *dfx;
    cudaMalloc(&dfx,sizeof(double)*n);

    //check QR result
    double normA = dnorm(m,n,A);
    dgemm(m,n,n,dA,m,R,n,A,m,1.0,-1.0);
    double normResult = dnorm(m,n,A);
    printf("||A-QR||/||A|| is %.3e\n", normResult/normA);
    cudaFree(dA);

    sllsSolve(m,n,reflector, lda, d_tau, d_work, lwork, R, dfx);
    host2Device(m,n,A,hA);
    //printMatrixDevice("A.csv",m, n, A, m);
    //printMatrixDevice("x.csv",n, 1, dfx, n);
    checkResult(m, n, A, lda,dfx);
   
}