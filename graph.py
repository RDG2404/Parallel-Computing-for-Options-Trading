import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load call and put option prices from .txt files
option_prices_c = np.loadtxt('callprice.txt')
option_prices_p = np.loadtxt('putprice.txt')

# Define range of x values
x_range = np.linspace(-14, 45, len(option_prices_p))

# Define kernel function
def kernel_call(x, h):
    return np.exp(-0.5 * ((x - h) / h)**2) / np.sqrt(2 * np.pi * h**2)

def kernel_put(x, h):
    return np.exp(-0.5 * ((x + h) / h)**2) / np.sqrt(2 * np.pi * h**2)

# Define function to calculate risk-neutral density using Gaussian kernels
def risk_neutral_density(x, h_c, h_p):
    density_c = np.mean(kernel_call(x - option_prices_c, h_c)) / h_c
    density_p = np.mean(kernel_put(x - option_prices_p, h_p)) / h_p
    return density_c + density_p

# Set bandwidth parameters
h_c = 1.377
h_p = 1.078

# Calculate risk-neutral density
density = [risk_neutral_density(x, h_c, h_p) for x in x_range]
# peak_index=np.argmax(density)
# peak_x = x_range[peak_index]
# peak_y = density[peak_index]
# Plot density
plt.title('Risk-neutral density for VIX options price')
plt.plot(x_range+12, density, label='RND')
plt.axvline(x=15, color='red', linestyle='--', label='X = 15 USD')
plt.xlabel('Underlying Asset Price')
plt.ylabel('Density')
plt.legend()
plt.show()