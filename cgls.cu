#include <stdlib.h>
#include <cmath>

#include "cgls.cuh"


void test6(int m,int n,double* dA,double* dR,double *db,double *dx)
{
    //double hA[m*n],hb[m];
    //cudaMemcpy(hA, dA, m* n*sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(hb, db, m*sizeof(double), cudaMemcpyDeviceToHost);
    cublasHandle_t handle;
    cublasCreate(&handle);
    double *hx = (double*)malloc(sizeof(double)*n);
    for(int i = 0;i<n;i++)
    {
        hx[i] = 1.0;
    }
    double shift = 0;
    double tol = 1e-12;
    int maxit = 50;
    bool quiet = false;
    
    cudaMemcpy(dx, hx, n * sizeof(double), cudaMemcpyHostToDevice);
    int flag = cgls::cgSolve<double>(handle,dA, dR,m, n, db, dx, shift, tol, maxit, quiet);
    printf("flag = %d\n",flag);
    cudaMemcpy(hx, dx, n * sizeof(double), cudaMemcpyDeviceToHost);
    //printmatrixd("A.csv",m,n,hA,m);
    //printmatrixd("b.csv",m,1,hb,m);
    //printmatrixd("x.csv",n,1,hx,n);
    //printf("%lf %lf\n",hx[0],hx[1]);
    return;
}

void printmatrixd(char *filename, int m, int n, double* a, int lda)
{
    
    FILE *f = fopen(filename, "w");
    if (f==NULL) {
        printf("fault!\n");
        return;
    }
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            fprintf(f, "%.8lf", a[i+j*lda]);
            if (j==n-1) fprintf(f, "\n");
            else fprintf(f,",");
        }
    }
    
    
    fclose(f);
}

int readfile(char* path,double* mat,int d)
{
    
    FILE *fp=fopen(path,"r");
    if(fp == NULL)
    {
        return 1;
    }
    char temp[300000];
    int i,j;
    i=0;j=0;
    //printf("1\n");
    while(!feof(fp))
    {
        i=0;


        fgets(temp,300000,fp);

        temp[strlen(temp)-1]= '\0';

        char *s;

        s=strtok(temp,",");

        mat[i*d+j] = atof(s);
        while(i<=d)
        {
            i++;
            //printf("s is %s %d\n",s,s[0]);
            s = strtok(NULL,",");
            if(s==NULL)
                break;
            mat[i*d+j]=atof(s);
            //printf("3\n");
        }
        j++;
    }
    
    return 1;
}

int main(int argc, char *argv[])
{
    int m,n;
    if(argc == 3)
    {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }
    else
    {
        printf("Please input m and n!\n");
        return 0;
    }
    double *hA = (double*)malloc(sizeof(double)*m*n);
    double *hR = (double*)malloc(sizeof(double)*n*n);
    double hb[m],hx[n];
    readfile("A.csv",hA,m);
    readfile("R.csv",hR,n);
    printf("Here\n");
    for(int i=0;i<m;i++)
    {
        hb[i] = 1.0;
    }
    //printmatrixd("NewA.csv",m,n,hA,m);
    //printmatrixd("NewR.csv",n,n,hR,n);
    
    //initialization
    double negone = -1;
    double one = 1;
    double zero = 0;
    double *dA,*dR,*db,*dx;
    cudaMalloc(&dA,sizeof(double)*m*n);
    cudaMalloc(&dR,sizeof(double)*n*n);
    cudaMalloc(&db,sizeof(double)*m);
    cudaMalloc(&dx,sizeof(double)*n);
    cudaMemcpy(dA,hA,sizeof(double)*m*n,cudaMemcpyHostToDevice);
    cudaMemcpy(dR,hR,sizeof(double)*n*n,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,sizeof(double)*m,cudaMemcpyHostToDevice);
    //cudaMemcpy(dx,hx,sizeof(double)*n,cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    //solve
    test6(m,n,dA,dR,db,dx);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cgls takes %f milliseconds\n",milliseconds);
    //
    cudaMemcpy(dA,hA,sizeof(double)*m*n,cudaMemcpyHostToDevice);
    cudaMemcpy(dR,hR,sizeof(double)*n*n,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,sizeof(double)*m,cudaMemcpyHostToDevice);
    
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    using namespace cgls;
    cublasDtrsm(handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,n,1,&one,dR,n,dx,n);
    double normx;
    double *tempx;
    cudaMalloc(&tempx,sizeof(double)*n);
    nrm2(handle, n, dx, &normx);
    cublasDgemv(handle,CUBLAS_OP_T,m,n,&one,dA,m,db,1,&zero,tempx,1);
    double normb;
    nrm2(handle, n, tempx, &normb);
    cublasDgemv(handle,CUBLAS_OP_N,m,n,&one,dA,m,dx,1,&negone,db,1);
    cublasDgemv(handle,CUBLAS_OP_T,m,n,&one,dA,m,db,1,&zero,dx,1);

    double normsol,norma;
    nrm2(handle, m, db, &normsol);
    nrm2(handle, n*m, dA, &norma);
    printf("||Ax-b|| = %.32lf\n ||A||=%.32lf\n",normsol,norma);
    nrm2(handle, n, dx, &normsol);
    printf("||A^T*(Ax-b)||/(||A^T||*||b||) = %.6e\n",normsol/normb);
    return 0;
}
