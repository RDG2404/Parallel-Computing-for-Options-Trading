#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "rungpu.cuh"


int main(int argc,char *argv[])
{
	printf("Starting estimating ... \n");
	//
	double tau = 1.0/12.0;
	double r = 0.02;
	int ngrid = atoi(argv[1]);
        //
	run_in_gpu(0.75,2.0,ngrid,
		   0.75,2.0,ngrid,
		   10.0,47.5,128,
		   tau,r);
	//
	//
	return 0;
}