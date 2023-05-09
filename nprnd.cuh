#ifndef nprnd_cuh
#define nprnd_cuh

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "qp.cuh"
//
//
class NPRND{
protected:
    //
    int ncall;    // number of observations (call)
    double *callp;  // call prices
    double *calls;  // call strikes
    double *callw;  // call weights
    //
    int nput;    // number of observations (put)
    double *putp;  // put prices
    double *puts; // put strikes
    double *putw; // put weights
    //
    double r;    // interest risk-free rate
    double tau;  // time to maturiry
    //
    //
    //
    double *xRange;
    int mRange;     // number of grid elements
    //
    double *strike;   // vector of unique strikes in sample
    int nstrike;     // vector dimension
    //
    //
    // computational statistics
    unsigned long nqpproblems;
    unsigned long niterations;
    //
    double solqp[8];
    //
    double cv;
    double area;
    double entropy;
    double variation;
    //
    //
public:
    //
    __device__ NPRND(double *callp,double *calls,double *callw,int ncall,
          double *putp,double *puts,double *putw,int nput,
          double r,double tau,
          double *xRange,int mRange,
          double *strike,int nstrike)
    {
        //
        this->ncall = ncall;
	this->callp = callp;
	this->calls = calls;
	this->callw = callw;
        //
	//
        this->nput = nput;
	this->putp = putp;
	this->puts = puts;
	this->putw = putw;
        //
        //
        this->r = r;
        this->tau = tau;
        //
        this->mRange = mRange;
        this->xRange = xRange;
        //
        this->nstrike = nstrike;
	this->strike = strike;
        //
        //
        this->cv = 0.0;
        this->entropy = 0.0;
        this->area = 0.0;
        this->variation = 0.0;
	//
	//
    }
    //
    //
    __device__ ~NPRND()
    {
    }
    //
    //
    __device__ void estim(double x0,double hc,double hp)
    {
        //
        //
        double x11 = 0.0; double x12 = 0.0; double x13 = 0.0; double x14 = 0.0;
        double x22 = 0.0; double x23 = 0.0; double x24 = 0.0;
        double x33 = 0.0; double x34 = 0.0;
        double x44 = 0.0;
        //
        double xy1 = 0.0;
        double xy2 = 0.0;
        double xy3 = 0.0;
        double xy4 = 0.0;
        //
        for(int i=0;i<this->ncall;i++){
            double t0 = (this->calls[i] - x0);
            double tt = sqrt(this->callw[i]*exp(-0.5*t0*t0/hc/hc));
            double t1 = tt;
            double t2 = t1*t0;
            double t3 = 0.5*t2*t0;
            double t4 = (2.0/6.0)*t3*t0;
            x11 += t1*t1;
            x12 += t1*t2;
            x13 += t1*t3;
            x14 += t1*t4;
            x22 += t2*t2;
            x23 += t2*t3;
            x24 += t2*t4;
            x33 += t3*t3;
            x34 += t3*t4;
            x44 += t4*t4;
            xy1 += tt*this->callp[i]*t1;
            xy2 += tt*this->callp[i]*t2;
            xy3 += tt*this->callp[i]*t3;
            xy4 += tt*this->callp[i]*t4;
            this->niterations++;
        }
        double xp11 = 0.0; double xp12 = 0.0; double xp13 = 0.0; double xp14 = 0.0;
        double xp22 = 0.0; double xp23 = 0.0; double xp24 = 0.0;
        double xp33 = 0.0; double xp34 = 0.0;
        double xp44 = 0.0;
        //
        double xyp1 = 0.0;
        double xyp2 = 0.0;
        double xyp3 = 0.0;
        double xyp4 = 0.0;
        //
            for(int i=0;i<this->nput;i++){
                double t0 = (this->puts[i] - x0);
                double tt = sqrt(this->putw[i]*exp(-0.5*t0*t0/hp/hp));
                double t1 = tt;
                double t2 = t1*t0;
                double t3 = 0.5*t2*t0;
                double t4 = (2.0/6.0)*t3*t0;
                xp11 += t1*t1;
                xp12 += t1*t2;
                xp13 += t1*t3;
                xp14 += t1*t4;
                xp22 += t2*t2;
                xp23 += t2*t3;
                xp24 += t2*t4;
                xp33 += t3*t3;
                xp34 += t3*t4;
                xp44 += t4*t4;
                //
                xyp1 += tt*this->putp[i]*t1;
                xyp2 += tt*this->putp[i]*t2;
                xyp3 += tt*this->putp[i]*t3;
                xyp4 += tt*this->putp[i]*t4;
                this->niterations++;
            }
        //
        //
        double H[64];
        H[0] = x11;  H[1]  = x12; H[2]  = x13; H[3] =  x14; H[4] =  0.0;  H[5] =  0.0;  H[6]  = 0.0;  H[7] =  0.0;
        H[8] = x12;  H[9]  = x22; H[10] = x23; H[11] = x24; H[12] = 0.0;  H[13] = 0.0;  H[14] = 0.0;  H[15] = 0.0;
        H[16] = x13; H[17] = x23; H[18] = x33; H[19] = x34; H[20] = 0.0;  H[21] = 0.0;  H[22] = 0.0;  H[23] = 0.0;
        H[24] = x14; H[25] = x24; H[26] = x34; H[27] = x44; H[28] = 0.0;  H[29] = 0.0;  H[30] = 0.0;  H[31] = 0.0;
        H[32] = 0.0; H[33] = 0.0; H[34] = 0.0; H[35] = 0.0; H[36] = xp11; H[37] = xp12; H[38] = xp13; H[39] = xp14;
        H[40] = 0.0; H[41] = 0.0; H[42] = 0.0; H[43] = 0.0; H[44] = xp12; H[45] = xp22; H[46] = xp23; H[47] = xp24;
        H[48] = 0.0; H[49] = 0.0; H[50] = 0.0; H[51] = 0.0; H[52] = xp13; H[53] = xp23; H[54] = xp33; H[55] = xp34;
        H[56] = 0.0; H[57] = 0.0; H[58] = 0.0; H[59] = 0.0; H[60] = xp14; H[61] = xp24; H[62] = xp34; H[63] = xp44;
        //
        //
        double f[8];
        f[0] = -1.0*xy1;
        f[1] = -1.0*xy2;
        f[2] = -1.0*xy3;
        f[3] = -1.0*xy4;
        f[4] = -1.0*xyp1;
        f[5] = -1.0*xyp2;
        f[6] = -1.0*xyp3;
        f[7] = -1.0*xyp4;
        //
        double tt = exp(-1.0*this->r*this->tau);
        double A[64] = {1,0,0,0,0,0,0,0,
                        0,1,-1,0,0,0,0,0,
                        0,0,0,1,0,0,0,0,
                        0,0,0,0,0,0,0,0,
                        0,0,0,0,1,0,0,0,
                        0,0,0,0,0,1,1,0,
                        0,0,0,0,0,0,0,1,
                        0,0,0,0,0,0,0,0
                        };
        //
        double b[8] = {0,-1.0*tt,0,0,0,0,-1.0*tt,0};
        //
        double Aeq[24] =  {0,0,0,
                           0,0,-1,
                           0,-1,0,
                           -1,0,0,
                            0,0,0,
                            0,0,1,
                            0,1,0,
                            1,0,0};
        //
        double beq[3] = {0,0,tt};
        //
        QP qp = QP(H,f,8,A,b,8,Aeq,beq,3);
        qp.solve(this->solqp);
        this->nqpproblems++;
    }
    //
    //
    //
    __device__ void estimxex(double x0,double x1,double hc,double hp)
    {
            double x11 = 0.0; double x12 = 0.0; double x13 = 0.0; double x14 = 0.0;
            double x22 = 0.0; double x23 = 0.0; double x24 = 0.0;
            double x33 = 0.0; double x34 = 0.0;
            double x44 = 0.0;
            //
            double xy1 = 0.0;
            double xy2 = 0.0;
            double xy3 = 0.0;
            double xy4 = 0.0;
            //
            for(int i=0;i<this->ncall;i++){
                if(this->calls[i] != x1){
                    double t0 = (this->calls[i] - x0);
                    double tt = sqrt(this->callw[i]*exp(-0.5*t0*t0/hc/hc));
                    double t1 = tt;
                    double t2 = t1*t0;
                    double t3 = 0.5*t2*t0;
                    double t4 = (2.0/6.0)*t3*t0;
                    x11 += t1*t1;
                    x12 += t1*t2;
                    x13 += t1*t3;
                    x14 += t1*t4;
                    x22 += t2*t2;
                    x23 += t2*t3;
                    x24 += t2*t4;
                    x33 += t3*t3;
                    x34 += t3*t4;
                    x44 += t4*t4;
                    xy1 += tt*this->callp[i]*t1;
                    xy2 += tt*this->callp[i]*t2;
                    xy3 += tt*this->callp[i]*t3;
                    xy4 += tt*this->callp[i]*t4;
                    this->niterations++;
                }

            }
            double xp11 = 0.0; double xp12 = 0.0; double xp13 = 0.0; double xp14 = 0.0;
            double xp22 = 0.0; double xp23 = 0.0; double xp24 = 0.0;
            double xp33 = 0.0; double xp34 = 0.0;
            double xp44 = 0.0;
            //
            double xyp1 = 0.0;
            double xyp2 = 0.0;
            double xyp3 = 0.0;
            double xyp4 = 0.0;
            //
            for(int i=0;i<this->nput;i++){
                if(this->puts[i] != x1){
                    double t0 = (this->puts[i] - x0);
                    double tt = sqrt(this->putw[i]*exp(-0.5*t0*t0/hp/hp));
                    double t1 = tt;
                    double t2 = t1*t0;
                    double t3 = 0.5*t2*t0;
                    double t4 = (2.0/6.0)*t3*t0;
                    xp11 += t1*t1;
                    xp12 += t1*t2;
                    xp13 += t1*t3;
                    xp14 += t1*t4;
                    xp22 += t2*t2;
                    xp23 += t2*t3;
                    xp24 += t2*t4;
                    xp33 += t3*t3;
                    xp34 += t3*t4;
                    xp44 += t4*t4;
                    //
                    xyp1 += tt*this->putp[i]*t1;
                    xyp2 += tt*this->putp[i]*t2;
                    xyp3 += tt*this->putp[i]*t3;
                    xyp4 += tt*this->putp[i]*t4;
                    this->niterations++;
                }

            }

            double H[64];
            H[0] = x11;  H[1]  = x12; H[2]  = x13; H[3] =  x14; H[4] =  0.0;  H[5] =  0.0;  H[6]  = 0.0;  H[7] =  0.0;
            H[8] = x12;  H[9]  = x22; H[10] = x23; H[11] = x24; H[12] = 0.0;  H[13] = 0.0;  H[14] = 0.0;  H[15] = 0.0;
            H[16] = x13; H[17] = x23; H[18] = x33; H[19] = x34; H[20] = 0.0;  H[21] = 0.0;  H[22] = 0.0;  H[23] = 0.0;
            H[24] = x14; H[25] = x24; H[26] = x34; H[27] = x44; H[28] = 0.0;  H[29] = 0.0;  H[30] = 0.0;  H[31] = 0.0;
            H[32] = 0.0; H[33] = 0.0; H[34] = 0.0; H[35] = 0.0; H[36] = xp11; H[37] = xp12; H[38] = xp13; H[39] = xp14;
            H[40] = 0.0; H[41] = 0.0; H[42] = 0.0; H[43] = 0.0; H[44] = xp12; H[45] = xp22; H[46] = xp23; H[47] = xp24;
            H[48] = 0.0; H[49] = 0.0; H[50] = 0.0; H[51] = 0.0; H[52] = xp13; H[53] = xp23; H[54] = xp33; H[55] = xp34;
            H[56] = 0.0; H[57] = 0.0; H[58] = 0.0; H[59] = 0.0; H[60] = xp14; H[61] = xp24; H[62] = xp34; H[63] = xp44;
            //
            //
            double f[8];
            f[0] = -1.0*xy1;
            f[1] = -1.0*xy2;
            f[2] = -1.0*xy3;
            f[3] = -1.0*xy4;
            f[4] = -1.0*xyp1;
            f[5] = -1.0*xyp2;
            f[6] = -1.0*xyp3;
            f[7] = -1.0*xyp4;
            //
            double tt = exp(-1.0*r*tau);
            double A[64] = {1,0,0,0,0,0,0,0,
                            0,1,-1,0,0,0,0,0,
                            0,0,0,1,0,0,0,0,
                            0,0,0,0,0,0,0,0,
                            0,0,0,0,1,0,0,0,
                            0,0,0,0,0,1,1,0,
                            0,0,0,0,0,0,0,1,
                            0,0,0,0,0,0,0,0
                            };
            //
            double b[8] = {0,-1.0*tt,0,0,0,0,-1.0*tt,0};
            //
            double Aeq[24] =  {0,0,0,
                               0,0,-1,
                               0,-1,0,
                               -1,0,0,
                                0,0,0,
                                0,0,1,
                                0,1,0,
                                1,0,0};
            //
            double beq[3] = {0,0,tt};
            //
            QP qp = QP(H,f,8,A,b,8,Aeq,beq,3);
            qp.solve(this->solqp);
            this->nqpproblems++;
    }
    //
    //
    //
    __device__ void estimCVElements(double xex,double hc,double hp)
    {
	    double *ddf = (double*)malloc(mRange*sizeof(double));
            for(int i=0;i<this->mRange;i++){
                this->estimxex(this->xRange[i],xex,hc,hp);
                ddf[i] = solqp[2];
                this->niterations++;
            }
            this->area = areaEstim(this->xRange,ddf,this->mRange);
            this->entropy = entropyEstim(this->xRange,ddf,this->mRange);
            this->variation = varEstim(ddf,this->mRange);
            //
	    free(ddf);
	    //
            this->estimxex(xex,xex,hc,hp);
            double fcall = this->solqp[0];
            double fput = this->solqp[4];
            int nct = 0;
            double cvc = 0.0;
            for(int k=0;k<this->ncall;k++){
                if(this->calls[k] == xex){
                    nct++;
                    double err = this->callp[k] - fcall;
                    cvc += err*err;
                    this->niterations++;
                }
            }
           if(nct > 0) cvc /= (double)nct;
            //
            int npt = 0;
            double cvp = 0.0;
            for(int k=0;k<this->nput;k++){
                if(this->puts[k] == xex){
                    npt++;
                    double err = this->putp[k] - fput;
                    cvp += err*err;
                    this->niterations++;
                }
            }
            if(npt > 0) cvp /= (double)npt;
            this->cv = (cvc + cvp);
    }
    //
    //
    __device__ double matCVElements(double hc,double hp,int k)
    {
        this->estimCVElements(this->strike[k],hc,hp);
        return this->cv*this->variation + (1.0 + fabs(this->area - 1.0))/this->entropy;
	//return this->cv*this->variation + log(1.0 + fabs(this->area -1.0)) - log(this->entropy);
    }

    __device__ double matCVElementsV1(double hc,double hp)
    {
	    double result = 0.0;
	    for(int k=0;k<this->nstrike;k++){
		    result += this->cv*this->variation + (1.0 + fabs(this->area - 1.0))/this->entropy;
	    }
	    return result;
    }
    
    //
    __device__ double get_cv()
    {
	    return this->cv;
    }
    //
    __device__ double get_variation()
    {
	    return this->variation;
    }
    //
    __device__ double get_area()
    {
	    return this->area;
    }
    //
    __device__ double get_entropy()
    {
	    return this->entropy;
    }
//
private:
    //
    //
    __device__ double areaEstim(double *x0,double *yy,int n)
    {
        double sum = 0.0;
        for(int i=0;i<n;i++){
            if(i==0){
                    sum  += 0.5*(x0[i+1] - x0[i])*yy[i];
                }else if(i==(n-1)){
                    sum += 0.5*(x0[i] - x0[i-1])*yy[i];
                }else{
                    sum += 0.5*(x0[i+1] - x0[i-1])*yy[i];
                }
        }
        return sum;
    }
    //
    //
    __device__ double entropyEstim(double *x0,double *yy,int n)
    {
        double area = areaEstim(x0, yy, n);
        double *y0 = (double*)malloc(n*sizeof(double));
        for(int i=0;i<n;i++) y0[i] = yy[i]/area;
        double sum = 0.0;
        for(int i=0;i<n;i++){
            if(i==0){
                    if(y0[i] > 0.0001) sum  += 0.5*(x0[i+1] - x0[i])*y0[i]*log(y0[i]);
                }else if(i==(n-1)){
                    if(y0[i] > 0.0001) sum += 0.5*(x0[i] - x0[i-1])*y0[i]*log(y0[i]);
                }else{
                    if(y0[i] > 0.0001) sum += 0.5*(x0[i+1] - x0[i-1])*y0[i]*log(y0[i]);
                }
        }
        free(y0);
        return -1.0*sum;
    }
    //
    //
    __device__ double varEstim(double *yy,int n)
    {
        double sum = 0.0;
        for(int i=1;i<n;i++){
            sum += fabs(yy[i] - yy[i-1]);
        }
        return sum;
    }

};
//
#endif /* rnd_hpp */