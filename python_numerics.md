```python
import numpy as np
"""
Generating numbers in numpy array
"""
# Giving the seed of random number generator
np.random.seed(1)
# Generate a 20*20 matrix with uniformly distributed random integers between 0 to 50
A = np.random.randint(low=0, high=50, size=(20, 20))
print(A)
```

```python
import matplotlib.pyplot as plt
fig = plt.figure() # Create a window to draw
ax = plt.axes() # Create axes
ax.plot(np.arange(1,21), A, 'o') # Plot the dots, using circle
ax.set_title('Random Number Plotting') # Set figure title
plt.show()
# Save figure to a high quality
fig.savefig('random_scatter.png', dpi=300)
fig.clear() # clear the figure

# Another plot: a time series
x = np.random.randn(1000) # 1000 samples of N(0,1)
# plt also provide a quick way to draw
plt.plot(x,'.') # Try other styles like ‘+’
plt.title('Normal Distribution White Noise')
plt.xlabel('i') # Putting label on the axes
plt.ylabel('x_i')
plt.draw()
plt.savefig('normal_dis_scatter.png', dpi=300)
```

```python
"""
Matrix operations
"""
# [[3, 1],[1, 1]] would create a 2x2 matrix
# Inverse matrix
A_inv = np.linalg.inv(A)
# Matrix multiplication operation
dot_result = np.dot(A, A_inv)
print(dot_result)
# Generate a 20*20 identity matrix
idn_matrix = np.identity(20)
print(idn_matrix)
# Using .allclose() function to evaluate two matrices are equal within tolerance
np.allclose(dot_result, np.eye(20)) # True
```

```python
"""
Matrix operations
"""
# [[3, 1],[1, 1]] would create a 2x2 matrix
A = [[3, 1],[1, 1]]
A_eig = np.linalg.eig(A)
# Now the Jordan decomposition A = Gamma*Lambda*Gamma^T
# Lambda is the diagnal matrix of the eig 
# Gamma is the vector 
E_val = A_eig[0]
Gamma = A_eig[1]
Lambda = np.diag(E_val)
# Check the result, you might get something within numerical eps
AA = np.dot( np.dot(Gamma, Lambda), np.transpose(Gamma) )
print(AA)
AA == A # True 
print( np.allclose(AA, A) ) # True
# Calculation of there square root of A
Lambda12 = np.sqrt(Lambda)
A12 = np.dot( np.dot(Gamma, Lambda12), np.transpose(Gamma) )
```

```python

# Image Transform
from PIL import Image
from numpy.fft import fft,ifft
import numpy as np
# Open the image by using Python Imaging Library(PIL)
image_before=Image.open('berlin_view.jpg')
# Decoding and encoding image to float number
image_int=np.fromstring(image_before.tobytes(), dtype=np.int8)
# Processing Fourier transform
fft_transformed=fft(image_int)
# Filter the lower frequency, i.e. employ a high pass
fft_transformed=np.where(np.absolute(fft_transformed) < 9e4,0,fft_transformed)
# Inverse Fourier transform
fft_transformed=ifft(fft_transformed)
# Keep the real part
fft_transformed=np.int8(np.real(fft_transformed))
# Output the image
image_output=Image.frombytes(image_before.mode, image_before.size,
fft_transformed)
image_output.show()
```

