import numpy as np
from optboosting import OptBoosting
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# Generate a toy datasets for classification
x, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, random_state=42)
y = 2 * y - 1  # Make labels +1 or -1
x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.5, random_state=42)

# Define and fit models
pb = OptBoosting(loss='hinge', descent='proximal', n_estimators=100)  # Proximal boosting
pb.fit(x_train, y_train)  # Fit the model

apb = OptBoosting(loss='hinge', descent='proximal', n_estimators=15, fast=True)  # Accelerated proximal boosting
apb.fit(x_train, y_train)  # Fit the model

# Print accuracies
print('Validation accuracy (pb):', np.mean(y_valid == pb.predict(x_valid)))
print('Validation accuracy (apb):', np.mean(y_valid == apb.predict(x_valid)))
