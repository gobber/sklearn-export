# -*- coding: utf-8 -*-

from sklearn_export.estimator.regressor.Regressor import Regressor


class MLPRegressor(Regressor):
    """
    See also
    --------
    sklearn.neural_network.MLPRegressor

    http://scikit-learn.org/stable/modules/generated/
    sklearn.neural_network.MLPRegressor.html
    """

    # @formatter:on

    def __init__(self, estimator, **kwargs):
        """
        Port a trained estimator to a dict.

        Parameters
        ----------
        :param estimator : MLPRegressor
            An instance of a trained MLPRegressor estimator.
        """
        super(MLPRegressor, self).__init__(estimator, **kwargs)

        # Activation function ('identity', 'logistic', 'tanh' or 'relu'):
        if estimator.activation not in self.hidden_activation_functions:
            raise ValueError(("The activation function '%s' of the estimator "
                              "is not supported.") % estimator.activation)

        # Estimator:
        est = estimator

        self.output_activation = est.out_activation_
        self.hidden_activation = est.activation

        self.n_layers = est.n_layers_
        self.n_hidden_layers = est.n_layers_ - 2

        self.n_inputs = len(est.coefs_[0])
        self.n_outputs = est.n_outputs_

        self.hidden_layer_sizes = est.hidden_layer_sizes
        if isinstance(self.hidden_layer_sizes, int):
            self.hidden_layer_sizes = [self.hidden_layer_sizes]
        self.hidden_layer_sizes = list(self.hidden_layer_sizes)

        self.layer_units = \
            [self.n_inputs] + self.hidden_layer_sizes + [est.n_outputs_]

        # Weights:
        self.coefficients = est.coefs_

        # Bias:
        self.intercepts = est.intercepts_

    @property
    def hidden_activation_functions(self):
        """Get list of supported activation functions for the hidden layers."""
        return ['relu', 'identity', 'tanh', 'logistic']

    def load_model_data(self, model_data=None):

        if model_data is None:
            model_data = {}

        if 'type' not in model_data:
            model_data['type'] = ''

        model_data['layers'] = [int(l) for l in list(self._get_activations())]
        model_data['bias'] = self._get_intercepts()
        model_data['hidden_activation'] = self.hidden_activation
        model_data['output_activation'] = self.output_activation
        model_data['type'] += 'MLPRegressor'

        weights = []
        numrows = []
        numcolumns = []
        for c in self.coefficients:
            w = []
            for j in range(0, len(c[0])):
                for i in range(0, len(c)):
                    w.append(c[i][j])
            numrows.append(len(c))
            numcolumns.append(len(c[0]))
            weights.append(w)
        model_data['weights'] = weights
        model_data['numRows'] = numrows
        model_data['numColumns'] = numcolumns

        return model_data

    def _get_intercepts(self):
        """
        Create a list of interceptors.
        """
        return [i.tolist() for i in self.intercepts]

    def _get_activations(self):
        """
        Concatenate the layers sizes of the regressor except the input layer.
        """
        return [str(x) for x in self.layer_units[1:]]
