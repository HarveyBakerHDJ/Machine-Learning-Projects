# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Import Libraries

# +
# %matplotlib inline

from ipywidgets import interactive
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# -

# ## Function Main

# #### Defining parameters

# +
# / z function block
def z_function(x, y): # defines our z function and to take two arguments
    return np.sin(5 * x) * np.cos(5 * y) / 5 # returns a complex function of x and y

# / calculate gradient block
def calculate_gradient(x, y): # defines calculate gradient to take two arguments
    return np.cos(5 * x) * np.cos(5 * y), -np.sin(5 * x) * np.sin(5 * y) # returns derivatives with respect to both x and y 

# / generate example x and y values
x = np.arange(-1, 1, 0.05) # creates a numpy array of values in 0.05 step sizes from -1 to 1 
y = np.arange(-1, 1, 0.05) # creates a numpy array of values in 0.05 step sizes from -1 to 1 

# / creating coordinates grid
X, Y = np.meshgrid(x, y) # outputs a new array where each column in our inputted 1D array

# / Z value
Z = z_function(X, Y) # creates new Z variable calling the z_function function onto our 2D X, Y meshgrid

# / intialize current position / co - ordinates
current_pos = (0.7, 0.4, z_function(0.7, 0.4)) 

# / intialize hyper parameter
learning_rate = 0.01 # controls the step size 'we' move in 
# -

# ## Visualising Gradient Descent

# +
# / figure layout block
fig = plt.figure() # creates a matplotlib figure as 'fig'
ax = fig.add_subplot(111, projection='3d') # adds a 3d subplot to 'fig'

# / plotting surface
ax.plot_surface(X, Y, Z, cmap='viridis', zorder=0) # takes arguments X, Y, Z and specfies styling

# / adding scatter plot
scat = ax.scatter(current_pos[0], current_pos[1], current_pos[2], color='magenta', zorder=1) # creates our 'ball' 

# / gradient descent loop
for _ in range(1000): # runs 1000 times
    X_derivative, Y_derivative = calculate_gradient(current_pos[0], current_pos[1]) # computes the partial derivatives (gradient) with respect to X and Y
    X_new, Y_new = current_pos[0] - learning_rate * X_derivative, current_pos[1] - learning_rate * Y_derivative # updates the current position by moving in opposite direction of the gradient 
    current_pos = (X_new, Y_new, z_function(X_new, Y_new)) # sets the new positions co-ordinates by calling the z_function onto the new X and Y co-ordinates

    scat.remove()
    scat = ax.scatter(current_pos[0], current_pos[1], current_pos[2], color='magenta', zorder=1)
    
    plt.draw()
    plt.pause(0.01)

plt.show()
# -


