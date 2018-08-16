import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF
from sklearn.model_selection import train_test_split, ParameterGrid
from multiprocessing import Pool


# Initial estimators
class InitialEstimator(object):
    def __init__(self):
        self.val = None

    def predict(self, x):
        """Return prediction."""
        return self.val * np.ones(np.asarray(x).shape[0])


class MeanEstimator(InitialEstimator):
    """Estimator of the mean of data."""
    def fit(self, _, y):
        """Fit the parameter val."""
        self.val = np.mean(y, axis=0)
        return self


class QuantileEstimator(InitialEstimator):
    """Estimator of the tau-quantile of data."""
    def __init__(self, level=0.5):
        self.level = level
        super(QuantileEstimator, self).__init__()
        # self.val = None

    def fit(self, _, y):
        """Fit the parameter val."""
        self.val = np.percentile(y, 100 * self.level)
        return self


class HingeEstimator(InitialEstimator):
    """Initial estimator for the hinge loss."""
    def fit(self, _, y):
        """Fit the parameter val."""
        self.val = np.sign(np.sum(y))
        return self


class LogitEstimator(InitialEstimator):
    """Initial estimator for the logistic loss."""
    def fit(self, _, y):
        """Fit the parameter val."""
        nb_pos = np.sum((y == y.max()))
        nb_neg = y.size - nb_pos
        if nb_neg == 0:
            raise ValueError('There is no negative labels.')
        self.val = np.log(nb_pos / nb_neg)
        return self


class ExpEstimator(InitialEstimator):
    """Initial estimator for the exponential loss."""
    def __init__(self, beta=1.):
        self.beta = beta
        super(ExpEstimator, self).__init__()
        # self.val = None
    
    def fit(self, _, y):
        """Fit the parameter val."""
        nb_pos = np.sum((y == y.max()))
        nb_neg = y.size - nb_pos
        if nb_neg == 0:
            raise ValueError('There is no negative labels.')
        self.val = 1 / (2 * self.beta) * np.log(nb_pos / nb_neg)
        return self

    
# Loss classes
class Loss(object):
    @staticmethod
    def check_arguments(y, z):
        y = np.asarray(y)
        z = np.asarray(z)
        if z.ndim == 2:
            y = y.reshape(-1, 1)
        if y.shape[0] != z.shape[0]:
            raise ValueError('Arguments have different shapes.')
        return y, z


class RegressionLoss(Loss):
    def __init__(self):
        self.type = 'regression'


class ClassificationLoss(Loss):
    def __init__(self):
        self.type = 'classification'


class LeastSquaresLoss(RegressionLoss):
    """Loss object for least squares."""
    def __call__(self, y, z):
        """Compute the loss between y and z."""
        y, z = self.check_arguments(y, z)
        return 0.5 * np.mean((y - z)**2, axis=0)

    @property
    def init_estimator(self):
        """Initial estimator (scale parameter that minimizes the loss."""
        return MeanEstimator()

    def negative_gradient(self, y, z):
        """Negative gradient of the loss wrt z."""
        y, z = self.check_arguments(y, z)
        return y - z  # See below for "/ y.size"

    def line_search(self, y, z, t):
        """Line search for minimizing loss(y, z + gamma * t)."""
        y, z = self.check_arguments(y, z)
        y, t = self.check_arguments(y, t)
        sum_t = np.sum(t ** 2)
        if sum_t > 0:
            return np.sum((y - z) * t) / sum_t
        else:
            return 0

    def prox(self, coef, y, z):
        """Compute proximal operator."""
        y, z = self.check_arguments(y, z)
        return (coef * y + z) / (1 + coef)  # y.size discarded


class LeastAbsoluteLoss(RegressionLoss):
    """Loss object for least absolute deviations."""
    def __call__(self, y, z):
        """Compute the loss between y and z."""
        y, z = self.check_arguments(y, z)
        return np.mean(np.abs(y - z), axis=0)

    @property
    def init_estimator(self):
        """Initial estimator (scale parameter that minimizes the loss."""
        return QuantileEstimator(0.5)

    def negative_gradient(self, y, z):
        """Negative subgradient of the loss wrt z."""
        y, z = self.check_arguments(y, z)
        return np.sign(y - z)  # See below for "/ y.size"

    def line_search(self, y, z, t):
        """Line search for minimizing loss(y, z + gamma * t)."""
        break_points = np.unique((y - z)[t != 0] / t[t != 0])
        if len(break_points) > 0:
            u = np.outer(t, break_points)
            z, u = self.check_arguments(z, u)
            loss_values = self.__call__(y, z + u)
            return break_points[np.argmin(loss_values)]
        else:
            return 0

    def prox(self, coef, y, z):
        """Compute proximal operator."""
        y, z = self.check_arguments(y, z)
        return np.fmax(0, 1 - coef / (1e-15 + np.abs(z - y))) * (z - y) + y  # y.size discarded


class QuantileLoss(RegressionLoss):
    """Loss object for quantile regression."""
    def __init__(self, q=0.5):
        self.type = 'regression'
        self.q = q
        super(QuantileLoss, self).__init__()
        
    def __call__(self, y, z):
        y, z = self.check_arguments(y, z)
        return np.mean(np.fmax(self.q * (y - z), (self.q - 1) * (y - z)), axis=0)
        
    @property
    def init_estimator(self): 
        """Initial estimator (scale parameter that minimizes the loss)."""
        return QuantileEstimator(self.q)
        
    def negative_gradient(self, y, z):
        """Negative subgradient of the loss wrt z."""
        y, z = self.check_arguments(y, z)
        return (y > z) * self.q + (y < z) * (self.q - 1)  # See below for "/ y.size"
        
    def line_search(self, y, z, t):
        """Line search for minimizing loss(y, z + gamma * t)."""
        break_points = np.unique((y - z)[t != 0] / t[t != 0])
        if len(break_points) > 0:
            u = np.outer(t, break_points)
            z, u = self.check_arguments(z, u)
            loss_values = self.__call__(y, z + u)
            return break_points[np.argmin(loss_values)]
        else:
            return 0
        
    def prox(self, coef, y, z):
        """Compute proximal operator."""
        return y + (y - z < coef * (self.q - 1)) * (z - y + coef * (self.q - 1)) + (y - z > coef * self.q) *\
               (z - y + coef * self.q)  # y.size discarded


class HingeLoss(ClassificationLoss):
    """Hinge loss object for classification."""
    def __call__(self, y, z):
        y, z = self.check_arguments(y, z)
        return np.mean(np.fmax(0, 1 - (y * z)), axis=0)
    
    @property
    def init_estimator(self):
        """Initial estimator (scale parameter that minimizes the loss)."""
        return HingeEstimator()
    
    def negative_gradient(self, y, z):
        """Negative subgradient wrt z."""
        y, z = self.check_arguments(y, z)
        return (y*z < 1) * y  # We omit "/ y.size" because the line search is very scale sensitive
    
    def line_search(self, y, z, t):
        """Line search for minimizing loss(y, z + gamma * t)."""
        mask = t != 0  # y should be -1 or +1
        break_points = np.unique((1 - y * z)[mask] / (y * t)[mask])

        if len(break_points) > 0:
            u = np.outer(t, break_points)
            z, u = self.check_arguments(z, u)
            loss_values = self.__call__(y, z + u)
            return break_points[np.argmin(loss_values)]
        else:
            return 0
    
    def prox(self, coeff, y, z):
        """Compute proximal operator."""
        y, z = self.check_arguments(y, z)
        return y + (y*z > 1) * (z - y) + (y*z < 1-coeff) * (z - y + y * coeff)


class LogitLoss(ClassificationLoss):
    """Logistic loss object for classification."""
    def __init__(self, it_newton=10):
        self.it_newton = it_newton
        super(LogitLoss, self).__init__()

    def __call__(self, y, z):
        y, z = self.check_arguments(y, z)
        return np.mean(np.log2(1 + np.exp(-y * z)))
    
    @property
    def init_estimator(self):
        return LogitEstimator()
        
    def negative_gradient(self, y, z):
        """Negative gradient wrt z."""
        y, z = self.check_arguments(y, z)
        return (y / (np.log(2))) * (1 - 1/(1 + np.exp(-y * z)))

    def line_search(self, y, z, t):
        """Line search for minimizing loss(y, z + gamma * t).
        This is one step of Newton-Raphson method from 0."""
        y, z = self.check_arguments(y, z)
        y, t = self.check_arguments(y, t)

        l = np.exp(-y*z)
        num = np.sum(y*t*l / (1+l))
        den = np.sum(t**2 * l / (1+l)**2)

        if den < 1e-10:
            return 0
        else:
            return num / den
    
    def prox(self, coeff, y, z):
        """Compute proximal operator."""
        y, z = self.check_arguments(y, z)

        pn = np.zeros(y.size)
        fact = coeff / np.log(2)
        for it in range(self.it_newton):
            e = np.exp(-y*pn)
            pn += (z - pn + fact * y*e / (1+e)) / (1 + fact * e / (1+e)**2)

        return pn


class ExpLoss(ClassificationLoss):
    """Exponential loss object for classification."""
    def __init__(self, beta=1., it_newton=10):
        self.beta = beta
        self.it_newton = it_newton
        super(ExpLoss, self).__init__()
        
    def __call__(self, y, z):
        y, z = self.check_arguments(y, z)
        return np.mean(np.exp(-self.beta * y * z))

    @property
    def init_estimator(self):
        """Initial estimator (scale parameter that minimizes the loss)."""
        return ExpEstimator(self.beta)
        
    def negative_gradient(self, y, z):
        """Negative gradient wrt z."""
        y, z = self.check_arguments(y, z)
        return self.beta * y * np.exp(-self.beta * y * z)  # Forget "/ y.size" because the Newton-Raphson step is very
        # sensitive to scale
    
    def line_search(self, y, z, t):
        """Line search for minimizing loss(y, z + gamma * t).
        This is one step of Newton-Raphson method from 0."""
        y, z = self.check_arguments(y, z)
        y, t = self.check_arguments(y, t)

        l = np.exp(-self.beta*y*z)
        den = self.beta * np.sum(t**2 * l)
        if den < 1e-10:
            return 0
        else:
            return np.sum(y*t*l) / den

    def prox(self, coeff, y, z):
        """Compute proximal operator.
        This is self.it_newton steps of Newton-Raphson method from 0."""
        y, z = self.check_arguments(y, z)

        pn = np.zeros(y.size)
        for it in range(self.it_newton):
            e = np.exp(-self.beta * y * pn)
            pn += (z - pn + coeff * self.beta * y * e) / (1 + coeff * self.beta**2 * e)

        return pn


LOSSES = {'ls': LeastSquaresLoss(),
          'lad': LeastAbsoluteLoss(),
          'quantile': QuantileLoss(),
          'hinge': HingeLoss(),
          'deviance': LogitLoss(),
          'exponential': ExpLoss()}


DESCENT_MODES = ['gradient', 'proximal']


# Optimization boosting model
class OptBoosting(BaseEstimator):
    """Accelerated gradient and proximal boosting.

    Accelerated gradient (proximal) boosting builds an additive model in a forward stage-wise fashion using Nesterov's
    acceleration; it allows for the optimization of arbitrary differentiable (proximable) loss functions.
    At each stage, a base learner is fitted on the negative gradient (or proximal direction) of the given loss function.

    Parameters
    ----------
    loss : {'ls', 'lad', 'quantile', 'hinge', 'deviance', 'exponential'}, optional (default='ls')
        Loss function to be optimized. Regression losses are: ls (least squares), lad (least absolute deviations) and
        quantile (pinball loss). Classification losses are: hinge, deviance (logit loss) and exponential.

    descent : {'gradient', 'proximal'}, optional (default='gradient')
        Optimization technique to decrease the loss.

    fast : bool, optional (default=False)
        Enables Nesterov's acceleration when it is true.

    n_estimators : int (default=100)
        Number of iterations to perform (equivalently, the number of estimators - 1 to aggregate).

    learning_rate : float, optional (default=0.1)
        Learning rate shrinks the contribution of each base estimator by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    step : float, optional (default=1.)
        Step coefficient for the proximal point algorithm.
        For a proximal descent, there is a trade-off between step, learning_rate and n_estimators.

    quantile : float, optional (default=0.5)
        Quantile level to estimate with the pinball loss. Should be between 0 and 1 excluded.

    beta : float, optional (default=1.)
        Parameter of the exponential loss.

    base_estimator : sklearn regressor, optional (default=DecisionTreeRegressor(criterion='friedman_mse', max_depth=3,
    presort=True))
        Family of base learners aggregated by boosting.

    tree_boosting : bool, optional (default=True)
        Enables one linesearch per leaf (only for regression tree as base estimator).

    train_size : float, optional (default=0.5)
        Ratio of the training size for fit_estim. The rest is used as a validation set to select the number of
        iterations.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If RandomState instance, random_state is
        the random number generator. If None, the random number generator is the RandomState instance used by np.random.

    refit : bool, optional (default=False)
        When fit_estim is used, the model is fitted with n_estimators minimizing the validation loss, with the subset of
        data defined for training.
        If refit is set to True, then the model is refitted with all data (training and validation).

    it_newton : int, optional (default=10)
        Number of Newton-Raphson iterations for computing proximal point (exponential and logistic loss).

    verbose : bool, optional (default=False)
        Print iteration number and objective function.

    Attributes
    ----------
    estimators : list, size = [n_estimators]
        List of estimators to aggregate.

    weights : list, size = [n_estimators]
        List of estimator weights.

    obj : list, size = [n_estimators]
        List of empirical loss values according to the iterations.

    obj_valid : list, size = [n_estimators]
        List of empirical loss values on the validation set (if specified) according to the iterations.

    loss_ : object, callable
        Loss object corresponding to the string loss.

    References
    ----------
    Accelerated gradient boosting (2018), G. Biau, B. Cadre, L. Rouvière. arXiv:1803.02042.
    Accelerated proximal boosting (2018), E. Fouillen, C. Boyer, M. Sangnier. hal-01853244.
    """
    def __init__(self, loss='ls', descent='gradient', fast=False, n_estimators=100, learning_rate=0.1, step=1.,
                 quantile=0.5, beta=1., base_estimator=None, tree_boosting=True, train_size=0.5, random_state=None,
                 refit=False, it_newton=10, verbose=False):
        self.loss = loss
        self.loss_ = None
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.fast = fast
        self.descent = descent
        self.step = step
        self.tree_boosting = tree_boosting
        self.estimators = []
        self.weights = []
        self.obj = []
        self.x_valid = []
        self.y_valid = None
        self.obj_valid = None
        self.train_size = train_size
        self.random_state = random_state
        self.refit = refit
        self.save_weights = False
        self.weights_iter = []
        self.it_newton = it_newton
        self.beta = beta
        self.quantile = quantile
        self.verbose = verbose
        self.best_n_estimators = None
        self.best_obj_valid = None

    def fit(self, x, y):
        """Fit the gradient boosting model.

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values (integers in classification, real numbers in regression).
            For classification, labels must correspond to classes.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check
        x = np.asarray(x, dtype=np.float32)
        if x.ndim < 2:
            x = x.reshape(-1, 1)

        # Compute validation objective?
        compute_valid = (self.x_valid is not None) and (self.y_valid is not None)

        # Initialization
        if self.loss in LOSSES:
            if self.loss == 'exponential':
                loss = ExpLoss(beta=self.beta, it_newton=self.it_newton)
            elif self.loss == 'deviance':
                loss = LogitLoss(it_newton=self.it_newton)
            elif self.loss == 'quantile':
                loss = QuantileLoss(q=self.quantile)
            else:
                loss = LOSSES[self.loss]
        else:
            raise ValueError('Unknown loss. Choose among:', LOSSES.keys())

        # Check classification labels
        if loss.type == 'classification':
            if not np.all(np.fabs(y) == 1):
                raise ValueError('Labels should be +1 and -1.')

        if self.descent not in DESCENT_MODES:
            raise ValueError('Unknown descent mode. Choose among:', DESCENT_MODES)

        self.estimators = [loss.init_estimator.fit(x, y)]  # List of estimators
        weights = np.zeros(self.n_estimators)  # Array of estimator weights
        weights[0] = 1
        if self.save_weights:
            self.weights_iter = [weights[:1]]

        # Default values for base_estimator and its parameters
        if self.base_estimator is None:
            self.base_estimator = DecisionTreeRegressor(criterion='friedman_mse', max_depth=3)
        # Speed up tree fitting
        if isinstance(self.base_estimator, DecisionTreeRegressor):
            self.base_estimator.set_params(presort=True)
            X_idx_sorted = np.asfortranarray(np.argsort(x, axis=0), dtype=np.int32)
        else:
            X_idx_sorted = None

        # Optimization variables
        pred = weights[0] * self.estimators[0].predict(x)  # Predictions
        interpol = pred.copy()  # Interpolation for acceleration
        coeff = 1  # Acceleration coefficient
        interpol_coeff = 0  # Interpolation coefficient

        self.obj = [loss(y, pred)]  # List of loss values
        if compute_valid:
            pred_valid = weights[0] * self.estimators[0].predict(self.x_valid)  # Predictions for validation set
            interpol_valid = pred_valid.copy()  # Interpolation for acceleration
            self.obj_valid = [loss(self.y_valid, pred_valid)]  # List of loss values for validation set

        # Loop
        for it in range(self.n_estimators-1):
            if self.verbose and it % 100 == 0:
                print('{0}: {1:0.4f}'.format(it, self.obj[-1]))
            # Pseudo-residues at the interpolated point
            if self.descent == 'gradient':
                residues = loss.negative_gradient(y, interpol)
            else:
                residues = (loss.prox(self.step, y, interpol) - interpol) / self.step

            estimator = clone(self.base_estimator)  # New estimator
            if X_idx_sorted is not None:
                estimator.fit(x, residues, check_input=False, X_idx_sorted=X_idx_sorted)
            else:
                estimator.fit(x, residues)
            grad = estimator.predict(x)  # Approximate direction of descent

            if isinstance(estimator, DecisionTreeRegressor) and self.tree_boosting:
                terminal_regions = estimator.tree_.apply(x)  # Leaf for each point in the dataset
                for leaf in np.where(estimator.tree_.children_left == TREE_LEAF)[0]:
                    ind = np.where(terminal_regions == leaf)[0]  # Points falling in that leaf
                    weight = loss.line_search(y[ind], interpol[ind], grad[ind])  # Line search
                    estimator.tree_.value[leaf, 0, 0] *= weight
                grad = estimator.predict(x)  # Approximate direction of descent with new leaf values
                weight = 1  # Weights are set to each leaf, so global weight boils down to 1
            else:
                weight = loss.line_search(y, interpol, grad)  # Line search
            weight *= self.learning_rate  # Shrinkage

            if self.fast:
                coeff_new = (1 + np.sqrt(1 + 4 * coeff**2)) / 2  # New acceleration coefficient
            else:
                coeff_new = 1
            interpol_coeff_new = (coeff - 1) / coeff_new  # Interpolation coeff

            pred_new = interpol + weight * grad  # New predictions
            interpol = pred_new + interpol_coeff_new * (pred_new - pred)  # Interpolation update

            if it > 0:
                new_weights = (weights - prev_weights) * (1 + interpol_coeff) + prev_weights  # Update weights
            else:
                new_weights = weights.copy()  # Keep weights at first iteration
            new_weights[it+1] = weight  # Add weight

            prev_weights = weights  # Previous weights update
            weights = new_weights  # Weights update
            interpol_coeff = interpol_coeff_new  # Interpolation coefficient update
            pred = pred_new  # Prediction update
            coeff = coeff_new  # Acceleration coefficient update

            self.estimators.append(estimator)  # Add estimator
            self.obj.append(loss(y, pred))  # Empirical loss

            if compute_valid:
                pred_valid_new = interpol_valid + weight * estimator.predict(self.x_valid)  # New predictions
                interpol_valid = pred_valid_new + interpol_coeff_new * (pred_valid_new - pred_valid)  # Interpol. update
                pred_valid = pred_valid_new

                self.obj_valid.append(loss(self.y_valid, pred_valid))  # List of loss values for validation set

            if self.save_weights:
                self.weights_iter.append(weights[:(it+2)])

        self.weights = list(weights)
        self.loss_ = loss

        # Check validity of weights update compared to iteration update
        pred_norm = np.sum(pred**2)
        err = np.sum((pred - self.decision_function(x))**2)
        err = err if pred_norm < 1e-10 else err / pred_norm  # Relative error
        if err > 1e-10:
            raise RuntimeWarning('Weights are different from what is expected (error: %e)' % err)

        return self

    def decision_function(self, x):
        """Compute the decision function for x.

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.

        Returns
        -------
        y : array-like, shape = [n_samples]
            Decision values.
        """
        # Check
        x = np.asarray(x)
        if x.ndim < 2:
            x = x.reshape(-1, 1)

        # Predictions
        return np.sum([w * estim.predict(x) for w, estim in zip(self.weights, self.estimators)], axis=0)

    def predict(self, x):
        """Predict regression target for x.

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.

        Returns
        -------
        y : array-like, shape = [n_samples]
            Predicted values (integers in classification, real numbers in regression).
        """

        # Predictions
        pred = self.decision_function(x)

        if self.loss_.type == 'classification':
            pred = np.sign(pred)
            pred[pred == 0] = 1

        return pred

    def score(self, x, y):
        """Compute the score (1 - loss) between y and predictions from x.

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values (integers in classification, real numbers in regression).
            For classification, labels must correspond to classes.

        Returns
        -------
        score : float
            The score is 1 - loss between y and predictions from x.
        """

        return 1 - self.loss_(self.predict(x), y)

    def loss_value(self, x, y):
        """Compute the loss between y and predictions from x.

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values (integers in classification, real numbers in regression).
            For classification, labels must correspond to classes.

        Returns
        -------
        loss : float
            Loss between y and predictions from x.
        """
        return self.loss_(self.predict(x), y)

    def set_valid(self, x, y):
        """Define data for computation of the validation loss, stored in self.obj_valid.

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values (integers in classification, real numbers in regression).
            For classification, labels must correspond to classes.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check
        x = np.asarray(x)
        if x.ndim < 2:
            x = x.reshape(-1, 1)

        self.x_valid = x.copy()
        self.y_valid = np.asarray(y).copy()
        return self

    def fit_estim(self, x, y, x_valid=None, y_valid=None):
        """Fit the gradient boosting model with n_estimators minimizing the validation loss.

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values (integers in classification, real numbers in regression).
            For classification, labels must correspond to classes.

        x_valid : array-like, shape = [n_samples_bis, n_features] (default=None)
            Validation vectors. If None, a ratio of size train_size is subsampled from x.

        y_valid : array-like, shape = [n_samples_bis] (default=None)
            Validation targets. If None, a ratio of size train_size is subsampled from y.

        Returns
        -------
        self : object
            Returns self.
        """
        has_valid = (x_valid is not None) and (y_valid is not None)
        self.save_weights = True

        if has_valid:
            self.set_valid(x_valid, y_valid)
            self.fit(x, y)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-self.train_size,
                                                                random_state=self.random_state)
            self.set_valid(x_test, y_test)
            self.fit(x_train, y_train)

        best_n_estimators = np.argmin(self.obj_valid) + 1
        if self.refit:
            n_estimators, self.n_estimators = self.n_estimators, best_n_estimators
            if has_valid:
                self.fit(np.concatenate((x, x_valid)), np.concatenate((y, y_valid)))
            else:
                self.fit(x, y)
            self.n_estimators = n_estimators
        else:
            self.weights = list(self.weights_iter[best_n_estimators-1])
            self.estimators = self.estimators[:best_n_estimators]
        self.best_n_estimators = best_n_estimators
        self.best_obj_valid = np.min(self.obj_valid)
        self.save_weights = False


def fit_grid_point(args):
    """Fit a model with a set of parameters.

    Parameters
    ----------
    args : tuple containing:
        estimator : estimator object
            This is assumed to implement the scikit-learn estimator interface.

        x : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values (integers in classification, real numbers in regression).
            For classification, labels must correspond to classes.

        x_valid : array-like, shape = [n_samples_bis, n_features] (default=None)
            Validation vectors. If None, a ratio of size train_size is subsampled from x.

        y_valid : array-like, shape = [n_samples_bis] (default=None)
            Validation targets. If None, a ratio of size train_size is subsampled from y.

        param : dict
            Dictionary with parameters names (string) as keys and associated values.

        loss : callable
            Loss function, taking as arguments true targest and predictions.

        verbose : bool
            Whether to print task number or not.

        i : int
            Task number

    Returns
    -------
    self : tuple
        Tuple containing the validation loss and the associated parameters.
    """
    estimator, x_train, y_train, x_valid, y_valid, param, loss, verbose, i = args
    est = clone(estimator).set_params(**param)

    if isinstance(est, OptBoosting):
        est.fit_estim(x_train, y_train, x_valid, y_valid)
        p = dict(n_estimators=est.best_n_estimators, **param)
    else:
        est.fit(x_train, y_train)
        p = param

    # Decision function for classifiers and predict for regressors
    if hasattr(est, 'decision_function'):
        pred = est.decision_function(x_valid)
    else:
        pred = est.predict(x_valid)

    if verbose:
        print(i, end=' ')

    return loss(y_valid, pred), p


def gridsearcheval(estimator, x_train, y_train, x_valid, y_valid, param_grid, loss, verbose=False):
    """Exhaustive search over specified parameter values for an estimator.
    This takes into account the fit_estim methods of OptBoosting objects.
    The parameters are selected by minimization of the loss on the validation set.

    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.

    x_train : array-like, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and n_features is the number of features.

    y_train : array-like, shape = [n_samples]
        Target values (integers in classification, real numbers in regression).
        For classification, labels must correspond to classes.

    x_valid : array-like, shape = [n_samples_bis, n_features] (default=None)
        Validation vectors. If None, a ratio of size train_size is subsampled from x.

    y_valid : array-like, shape = [n_samples_bis] (default=None)
        Validation targets. If None, a ratio of size train_size is subsampled from y.

    param_grid : dict
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values.

    loss : callable
        Loss function, taking as arguments true targest and predictions.

    verbose : bool (default: False)
        Whether to print task number or not.

    Returns
    -------
    s : tuple
        Tuple containing the minimum validation loss and the associated parameters.
    """
    param_list = ParameterGrid(param_grid)
    pool_args = [(estimator, x_train, y_train, x_valid, y_valid, param, loss, verbose, i)
                 for i, param in enumerate(param_list)]
    # res = [fit_grid_point(args) for args in pool_args]  # No multiprocessing
    if verbose:
        print('    tasks ({}):'.format(len(pool_args)), end=' ')
    with Pool() as p:
        res = p.map(fit_grid_point, pool_args)
    if verbose:
        print('')
    return min(res, key=lambda x: x[0])
