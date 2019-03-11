# -*- coding: utf-8 -*-

import os

from json import encoder
from json import dumps

from sklearn_export.estimator.classifier.Classifier import Classifier


class MLPClassifier(Classifier):
    """
    See also
    --------
    sklearn.neural_network.MLPClassifier

    http://scikit-learn.org/stable/modules/generated/
    sklearn.neural_network.MLPClassifier.html
    """

    # @formatter:on

    def __init__(self, estimator, **kwargs):
        """
        Port a trained estimator to the syntax of a chosen programming
        language.

        Parameters
        ----------
        :param estimator : MLPClassifier
            An instance of a trained MLPClassifier estimator.
        :param target_language : string, default: 'java'
            The target programming language.
        :param target_method : string, default: 'predict'
            The target method of the estimator.
        """

        super(MLPClassifier, self).__init__(estimator,  **kwargs)

        # Activation function ('identity', 'logistic', 'tanh' or 'relu'):
        hidden_activation = estimator.activation
        if hidden_activation not in self.hidden_activation_functions:
            raise ValueError(("The activation function '%s' of the estimator "
                              "is not supported.") % hidden_activation)

        # Output activation function ('softmax' or 'logistic'):
        output_activation = estimator.out_activation_
        if output_activation not in self.output_activation_functions:
            raise ValueError(("The activation function '%s' of the estimator "
                              "is not supported.") % output_activation)

        self.estimator = estimator

        # Estimator:
        est = self.estimator

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

        # Binary or multiclass classifier?
        self.is_binary = self.n_outputs == 1
        self.prefix = 'binary' if self.is_binary else 'multi'

    @property
    def hidden_activation_functions(self):
        """Get list of supported activation functions for the hidden layers."""
        return ['relu', 'identity', 'tanh', 'logistic']

    @property
    def output_activation_functions(self):
        """Get list of supported activation functions for the output layer."""
        return ['softmax', 'logistic']

    def load_model_data(self, model_data=None):

        if model_data is None:
            model_data = {}

        if 'type' not in model_data:
            model_data['type'] = ''

        model_data['layers'] = [int(l) for l in list(self._get_activations())]
        model_data['bias'] = [i.tolist() for i in self.intercepts]
        model_data['hidden_activation'] = self.hidden_activation
        model_data['output_activation'] = self.output_activation
        model_data['type'] += 'MLPBinaryClassifier' if self.is_binary else 'MLPMultiClassifier'

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

    def to_json(self, directory, filename, model_data=None, with_md5_hash=False):
        """
        Save model data in a JSON file.

        Parameters
        ----------
        :param directory : string
            The directory.
        :param filename : string
            The filename.
        :param with_md5_hash : bool, default: False
            Whether to append the checksum to the filename or not.
        """
        model_data = self.load_model_data(model_data=model_data)

        encoder.FLOAT_REPR = lambda o: self.repr(o)
        json_data = dumps(model_data, sort_keys=True)
        if with_md5_hash:
            import hashlib
            json_hash = hashlib.md5(str(json_data).encode('utf-8')).hexdigest()
            filename = filename.split('.json')[0] + '_' + json_hash + '.json'
        path = os.path.join(directory, filename)
        with open(path, 'w') as fp:
            fp.write(json_data)

    def _get_intercepts(self):
        """
        Concatenate all intercepts of the classifier.
        """
        temp_arr = self.temp('arr')
        for layer in self.intercepts:
            inter = ', '.join([self.repr(b) for b in layer])
            yield temp_arr.format(inter)

    def _get_activations(self):
        """
        Concatenate the layers sizes of the classifier except the input layer.
        """
        return [str(x) for x in self.layer_units[1:]]
