#ifndef rungpu_cuh
#define rungpu_cuh
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "io.cuh"
#include "kernelfunc.cuh"
//
//
//
__host__ __device__ void minMatrix(double *res,double *xxx,int nx,double *yyy,int ny,double *mat)
{
     double min = mat[0];
     int row = 0;
     int col = 0;
     for(int i=0;i<nx;i++){
         for(int j=0;j<ny;j++){
            if(mat[i*nx + j] < min){
               min = mat[i*nx + j];
               row = i;
               col = j;
           }
        }
   }
res[0] = xxx[row];
res[1] = yyy[col];
}
//
//

void run_in_gpu(double hcmin,double hcmax,int nhc,
		double hpmin,double hpmax,int nhp,
		double xmin,double xmax,int nx,
		double tau,double r)
{
int nc;
double *callprice = readArray2cuda("callprice.txt",&nc);
double *callstrike = readArray2cuda("callstrike.txt",&nc);
double *callopenint = readArray2cuda("callopenint.txt",&nc);
printf("Number observations (call) : %d\n",nc);
//
int np;
double *putprice = readArray2cuda("putprice.txt",&np);
double *putstrike = readArray2cuda("putstrike.txt",&np);
double *putopenint = readArray2cuda("putopenint.txt",&np);
printf("Number observations (put)  : %d\n",np);
//
int nu;
double *strike = readArray2cuda("strike.txt",&nu);
printf("Number of strikes: %d\n",nu);
//
double time_spent = 0.0;
clock_t begin = clock();
//
double *hc = linspaceCuda(hcmin,hcmax,nhc);
double *hp = linspaceCuda(hpmin,hpmax,nhp);
double *xRange = linspaceCuda(xmin,xmax,nx);
unsigned int size = nhc*nhp*nu;
double *out = createCudaArray(size);
//
cudaDeviceSetLimit(cudaLimitMallocHeapSize,1024*1024*4000L);
kernel<<<dim3(nhc,nhp),nu>>>(out,
				hc,nhc,hp,nhp,strike,nu,
                        	callprice,callstrike,callopenint,nc,
                        	putprice,putstrike,putopenint,np,
                        	xRange,nx,
				tau,r);
cudaDeviceSynchronize();
//
double *out_h = readArrayFromDevice(out,size);
double *cvvec = (double*)calloc(nhc*nhp,sizeof(double));
int ii = -1;
for(int k=0;k<nu;k++){
        int jj = -1;
        for(int i=0;i<nhc;i++){
                for(int j=0;j<nhp;j++){
                        ii++;
                        jj++;
                        cvvec[jj] += out_h[ii];
                }
        }
}
double *hc_h = linspace(hcmin,hcmax,nhc);
double *hp_h = linspace(hpmin,hpmax,nhp);
//
double sol[2];
minMatrix(sol,hc_h,nhc,hp_h,nhp,cvvec);
printf("hcoptim: %.4f\n",sol[0]);
printf("hpoptim: %.4f\n",sol[1]);
//
writeMatrix(cvvec,nhc,nhp,"CVMatrix.txt");
//
clock_t end = clock();
time_spent = (double)(end - begin)/CLOCKS_PER_SEC;
printf("The elapsed time is %f seconds\n",time_spent);
//
cudaDeviceReset();
//
free(cvvec);
free(hc_h);
free(hp_h);
//
printf("Done ... \n");
// memory free zone
cudaFree(callprice);
cudaFree(callstrike);
cudaFree(callopenint);
cudaFree(putprice);
cudaFree(putstrike);
cudaFree(putopenint);
cudaFree(strike);
cudaFree(hc);
cudaFree(hp);
cudaFree(out);
cudaFree(xRange);
free(out_h);
}
////////////////////////
#endif