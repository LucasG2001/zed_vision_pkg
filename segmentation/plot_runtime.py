import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

# Lists to store iteration number and segmentation time
iteration_numbers = []
segmentation_times = []

# Read data from the text file
with open(current_dir+'/segmentation_log.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        # Extract iteration number and segmentation time from each line
        parts = line.split('-')
        iteration_number = int(parts[0].strip().split()[-1])
        segmentation_time = float(parts[1].strip().split()[-2])  # Assuming the time is always the second-to-last element

        # Append the values to the lists
        iteration_numbers.append(iteration_number)
        segmentation_times.append(segmentation_time)

# Convert lists to numpy arrays
X = np.array(iteration_numbers).reshape(-1, 1)  # Reshape to a 2D array
y = np.array(segmentation_times)

# Perform linear regression
model = LinearRegression().fit(X, y)

# Predict segmentation times using the model
predictions = model.predict(X)

# Plotting the data and the linear regression line
plt.scatter(iteration_numbers, segmentation_times, marker='o', label='Actual Data')
plt.plot(iteration_numbers, predictions, color='red', label='Linear Regression')
plt.xlabel('Iteration Number')
plt.ylabel('Segmentation Time (seconds)')
plt.title('Segmentation Time vs Iteration Number with Linear Regression')
plt.legend()
plt.show()
