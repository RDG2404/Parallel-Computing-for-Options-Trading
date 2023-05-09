# This code visualizes the RND values using the calculated H_c and h_p bandwidths
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# We will generate two gaussian KDEs for call and put prices, then combine them over the strike price values
# Generate a range of asset prices, this will allow us to see the risk-neutral densities at each value of expected asset price

r=0.02 # Risk-free interest rate
tau=1.0/12.0 # Time to contract maturity
sigma= 0.2 # Volatility of asset

put_price=np.loadtxt('putprice.txt')
put_strike=np.loadtxt('putstrike.txt')
call_price=np.loadtxt('callprice.txt')
call_price= call_price[0:len(put_price)]
call_strike=np.loadtxt('callstrike.txt')
call_strike= call_strike[0:len(put_strike)]
strikes=np.loadtxt('strike.txt') # strikes sold data
S=np.linspace(min(strikes), max(strikes), len(put_price)) # Generates range of possible asset values from actual sale data
d1 = (np.log(S/put_strike)+(r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
d2 = d1 - sigma*np.sqrt(tau)
call_prob = stats.norm.pdf(d1) / (sigma * S * np.sqrt(tau))
put_prob = stats.norm.pdf(-d2) / (sigma * S * np.sqrt(tau))
risk_neutral_density = call_prob * call_price + put_prob * put_price

# Estimate RND using KDE with bandwidths obtained from Parallel computing implementation
h_c=1.3770
h_p=1.0778
call_density = stats.gaussian_kde(call_price, bw_method=h_c)
put_density = stats.gaussian_kde(put_price, bw_method=h_p)
estimated_density = 0.5 * call_density.evaluate(call_price) + 0.5 * put_density.evaluate(put_price)

# Plot the risk-neutral density
plt.title("VIX Risk-Neutral Density over Expected Asset Prices")
plt.plot(S, risk_neutral_density, label='Risk-Neutral Density')
plt.xlabel('Range of underlying asset price')
plt.ylabel('Density')
plt.legend()
plt.show()

