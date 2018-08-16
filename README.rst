.. -*- mode: rst -*-

optboosting
===========

optboosting is a Python library for boosting (based on a functional optimizaztion point of view), in particular based on gradient and proximal methods (referred to as gradient and proximal boosting). It also implements Nesterov's acceleration, enabling to build accurate models with less weak learners than traditional boosting.

Although a simple implementation of boosting, optboosting follows the `scikit-learn <http://scikit-learn.org>`_ style of programming.

Example
-------

.. code-block:: python

    import numpy as np
    from optboosting import OptBoosting
    from sklearn.datasets import make_classification

    # Generate a toy datasets for classification
    x, y = make_classification(n_samples=500, n_features=2, n_redundant=0, random_state=42)
    y = 2 * y - 1  # Make labels +1 or -1

    # Define and fit the accelerated proximal boosting model
    apb = OptBoosting(loss='hinge', descent='proximal', n_estimators=15, fast=True)
    apb.fit(x, y)

    # Prediction
    print('Decision values:', apb.decision_function(x)[:10])
    print('Predicted labels:', apb.predict(x)[:10])
    print('Accuracy:', np.mean(y == apb.predict(x)))

Dependencies
------------

optboosting needs Python >= 2.7, setuptools and Numpy.

Installation
------------

To install optboosting from pip, type::

    pip install https://github.com/msangnier/optboosting/archive/master.zip

To install optboosting from source, type::

    git clone https://github.com/msangnier/optboosting.git
    cd optboosting
    sudo python setup.py install

Authors
-------

Maxime Sangnier

References
----------

- Accelerated gradient boosting (2018), G. Biau, B. Cadre, L. Rouvière. arXiv:1803.02042.
- Accelerated proximal boosting (2018), E. Fouillen, C. Boyer, M. Sangnier. hal-01853244.
                                                                          
