import numpy as np
from optboosting import OptBoosting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Generate a toy dataset for regression
np.random.seed(42)

data = np.sort(np.random.rand(100))  # Input (train)
data_valid = np.sort(np.random.rand(data.size))  # Input (validation)

targets = np.sin(2 * np.pi * data) + np.random.randn(data.size) * 0.1  # Targets (train)
targets_valid = np.sin(2 * np.pi * data_valid) + np.random.randn(data.size) * 0.1  # Targets (validation)

data = data.reshape(-1, 1)  # Reshape as a column vector
data_valid = data_valid.reshape(-1, 1)  # Reshape as a column vector

# Base boosting model
reg = OptBoosting(loss='lad', n_estimators=100, learning_rate=1e-1, step=100)
reg.set_valid(data_valid, targets_valid)  # Set the validation set

# Figure
_, axes = plt.subplots(2, 2, sharex='col', figsize=(12, 10))

# Comparison gradient / proximal boosting
for i, opt in enumerate(['gradient', 'proximal']):
    # Plot data (left figure)
    axes[i, 0].plot(data, targets, label='Data')
    axes[i, 1].plot(0, 0)  # Blank point on the right figure for color consistency

    # Labels
    axes[i, 0].set_xlabel('x')
    axes[i, 0].set_ylabel('Prediction ({})'.format(opt))
    axes[i, 1].set_xlabel('Iteration')
    axes[i, 1].set_ylabel('Error ({})'.format(opt))

    for acceleration in [False, True]:
        reg.set_params(descent=opt, fast=acceleration)  # Define descent and acceleration modes
        reg.fit(data, targets)  # Fit the model

        # Plot prediction (left figure)
        axes[i, 0].plot(data, reg.predict(data),
                        label='Prediction ' + ('(accelerated)' if acceleration else '(vanilla)'))
        axes[i, 0].plot(0, 0)  # Blank point for color consistency

        # Plot errors (right figure)
        axes[i, 1].plot(reg.obj,
                        label='Training error ' + ('(accelerated)' if acceleration else '(vanilla)'))
        axes[i, 1].plot(reg.obj_valid, ':',
                        label='Validation error ' + ('(accelerated)' if acceleration else '(vanilla)'))

    # Legends
    axes[i, 0].legend()
    axes[i, 1].legend()

plt.show()
