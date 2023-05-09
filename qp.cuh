#ifndef qp_cuh
#define qp_cuh

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "ls.cuh"


/*
 Performs quadratic programming
 minimize 0.5*x' H x + x' f
 st.      A x > b
          C x = d
Syntax:
QP *qp = new QP(H,f,n,A,b,nineq,Aeq,beq,neq)
qp->solve(x)
Note: The matrices are inputed by column
    2 3
A = 5 6
    1 0
double *A = new double[6];
A[0] = 2; A[1] = 5; A[2] = 1;
A[3] = 3; A[4] = 6; A[5] = 0;
*/


class QP{
//
protected:
    double *H;
    double *f;
    int n;
    double *A;
    double *b;
    int nineq;
    double *Aeq;
    double *beq;
    int neq;
    //
    double *yy;
    double *xx;
//
public:
    __device__ QP(double *H,double *f,int n,double *A,double *b,int nineq)
    {
        this->n = n;
        this->H = H;
        this->f = f;
        this->A = A;
        this->b = b;
        this->nineq = nineq;
        this->Aeq = NULL;
        this->beq = NULL;
        this->neq = 0;
        this->yy = (double*)malloc(this->n*sizeof(double));
        this->xx = (double*)malloc(this->n*this->n*sizeof(double));
    }
    //
    __device__ QP(double *H,double *f,int n,double *A,double *b,int nineq,double *Aeq,double *beq,double neq)
    {
        this->n = n;
        this->H = H;
        this->f = f;
        this->A = A;
        this->b = b;
        this->nineq = nineq;
        this->Aeq = Aeq;
        this->beq = beq;
        this->neq = neq;
        this->yy = (double*)malloc(this->n*sizeof(double));
        this->xx = (double*)malloc(this->n*this->n*sizeof(double));
    }
    //
    //
    __device__ ~QP()
    {
        if(yy) free(yy);
        if(xx) free(xx);
    }
    //
    __device__ void solve(double *sol)
    {
        if(this->neq == 0)
        {
            qptols(this->H, this->f, this->yy, this->xx, this->n);
            LS ls = LS(this->yy,this->xx,this->n,this->n,this->A,this->b,this->nineq);
            ls.lsineq(sol);
        }else{
            qptols(this->H, this->f, this->yy, this->xx, this->n);
            LS ls = LS(this->yy,this->xx,this->n,this->n,this->A,this->b,this->nineq,this->Aeq,this->beq,this->neq);
            ls.lsineqeq(sol);
        }
    }
    //
    //
private:
    //
    __device__ void qptols(double *H,double *f,double *y,double *x,int n)
    {
        int size = n*n;
        chol(H,n);
        Lower_Triangular_Solve(H, f, y, n);
        for(int i=0;i<n;i++) y[i] = -1.0*y[i];
        for(int i=0;i<size;i++) x[i] = H[i];
    }
    //
    __device__ void chol(double *A,int n)
    {
        Choleski_LU_Decomposition(A,n);
        int iter = -1;
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                iter += 1;
                if(i<j){
                    A[iter] = 0.0;
                }
            }
        }
    }
    //
    __device__ int Choleski_LU_Decomposition(double *A, int n)
    {
        int i, k, p;
        double *p_Lk0;                   // pointer to L[k][0]
        double *p_Lkp;                   // pointer to L[k][p]
        double *p_Lkk;                   // pointer to diagonal element on row k.
        double *p_Li0;                   // pointer to L[i][0]
        double reciprocal;
        //
        for (k = 0, p_Lk0 = A; k < n; p_Lk0 += n, k++) {

            //            Update pointer to row k diagonal element.

            p_Lkk = p_Lk0 + k;

            //            Calculate the difference of the diagonal element in row k
            //            from the sum of squares of elements row k from column 0 to
            //            column k-1.

            for (p = 0, p_Lkp = p_Lk0; p < k; p_Lkp += 1,  p++)
                *p_Lkk -= *p_Lkp * *p_Lkp;

            //            If diagonal element is not positive, return the error code,
            //            the matrix is not positive definite symmetric.

            if ( *p_Lkk <= 0.0 ) return -1;

            //            Otherwise take the square root of the diagonal element.

            *p_Lkk = sqrt( *p_Lkk );
            reciprocal = 1.0 / *p_Lkk;

            //            For rows i = k+1 to n-1, column k, calculate the difference
            //            between the i,k th element and the inner product of the first
            //            k-1 columns of row i and row k, then divide the difference by
            //            the diagonal element in row k.
            //            Store the transposed element in the upper triangular matrix.

            p_Li0 = p_Lk0 + n;
            for (i = k + 1; i < n; p_Li0 += n, i++) {
                for (p = 0; p < k; p++)
                    *(p_Li0 + k) -= *(p_Li0 + p) * *(p_Lk0 + p);
                *(p_Li0 + k) *= reciprocal;
                *(p_Lk0 + i) = *(p_Li0 + k);
            }
        }
        return 0;
    }
    //
    __device__ int Lower_Triangular_Solve(double *L, double B[], double x[], int n)
    {
        int i, k;

        //         Solve the linear equation Lx = B for x, where L is a lower
        //         triangular matrix.

        for (k = 0; k < n; L += n, k++) {
            if (*(L + k) == 0.0) return -1;           // The matrix L is singular
            x[k] = B[k];
            for (i = 0; i < k; i++) x[k] -= x[i] * *(L + i);
            x[k] /= *(L + k);
        }

        return 0;
    }
    //
    //
};  // end of the QP Object

#endif /* qp_hpp */