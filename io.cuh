#ifndef io_cuh
#define io_cuh
//
//
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
//
//
__host__ void writeMatrix(double *x,int nl,int nc,const char* file)
{
	FILE *fp;
	fp = fopen(file,"wa");
	for(int i=0;i<nl;i++){
		for(int j=0;j<nc;j++){
			fprintf(fp,"%.4f ",x[nl*i + j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}
//
//
//
__host__ double* readArray2cuda(const char* file, int *n)
{
    FILE *fp;
    fp = fopen(file, "r");
    char buff[64];
    int nrow = -1;
    while (!feof(fp)) {
        fscanf(fp, "%s",buff);
        nrow++;
    }
    *n = nrow;
    rewind(fp);
    double *result = (double*)malloc(nrow*sizeof(double));
    for(int i=0;i<nrow;i++){
        fscanf(fp, "%s",buff);
        result[i] = atof(buff);
    }
    fclose(fp);
    double *x_d;
    cudaMalloc((void**)&x_d,nrow*sizeof(double));
    cudaMemcpy(x_d,result,nrow*sizeof(double),cudaMemcpyHostToDevice);
    free(result);
    return x_d;
}
//
//
//
__host__ double* readArrayFromDevice(double *x_d,unsigned int n)
{
	double *x = (double*)malloc(n*sizeof(double));
	cudaMemcpy(x,x_d,n*sizeof(double),cudaMemcpyDeviceToHost);
	return x;
}
//
//
__host__ double* createCudaArray(unsigned int n)
{
	double *x_d;
	cudaMalloc((void**)&x_d,n*sizeof(double));
	return x_d;
}
//
//
__host__ double* linspaceCuda(double xmin,double xmax,int n)
{
    double *xxx = (double*)malloc(n*sizeof(double));
    double step = (xmax - xmin)/(double)(n - 1);
    xxx[0] = xmin;
    for(int i=1;i<n;i++){
        xxx[i] = xxx[i-1] + step;
    }
    double *x_d;
    cudaMalloc((void**)&x_d,n*sizeof(double));
    cudaMemcpy(x_d,xxx,n*sizeof(double),cudaMemcpyHostToDevice);
    free(xxx);
    return x_d;
}
//
//
__host__ double* linspace(double xmin,double xmax,int n)
{
    double *xxx = (double*)malloc(n*sizeof(double));
    double step = (xmax - xmin)/(double)(n - 1);
    xxx[0] = xmin;
    for(int i=1;i<n;i++){
        xxx[i] = xxx[i-1] + step;
    }
    return xxx;
}
//
//
#endif