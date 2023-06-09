# Parallel-Computing-for-Options-Trading
This project aims to develop parallel computing algorithms in C++ using CUDA to quickly identify the optimal bandwidths for risk-neutral densities in options pricing for real-time financial risk assessment. 

A European-type call/put option is a contract giving the right, but not the obligation for the holder to purchase or sell a given stock S, at a predetermined price X, at the maturity date T. A central object in option pricing is the risk-neutral density (RND) which informs derivative security pricing at an expected value at the time of contract maturity. This would allow for lower (neutral) risk in negotiating option contracts as the risk profile of equities can be predicted and assessed prior to maturity. The expected values of the equities at the time of maturity can be calculated using the risk-neutral densitiy (RND) of the options prices.

We compare the processing times required for identifying the optimal kernel bandwidths in the non-parametric estimation of the RND for both sequential and parallel computing algorithms created using C++. SP500 and Volatility Index (VIX) values between April 18-20, 2018 are used with sample size of 9785 and 1456 respectively to generate the RNDs. Minimum grid search methods along with tailor-made Cross Validation (CV) algorithms are used to identify the optimal bandwidths for the same. 

We can see that as the sample size is increased, computational times exceed 48 hrs which would invalidate any results in the context of assessing real-time financial risk. For this reason, parallel computing algorithms have been developed in CUDA/C++ which are expected to achieve sppedups of upto 70x. 

This repository contains the CUDA code for the same, and was run using gcc compiler on NVIDIA Tesla V100 GPU, the file "Report_RND Options Trading.pdf" lists the methodology, results and conclusions of this project in detail. 
