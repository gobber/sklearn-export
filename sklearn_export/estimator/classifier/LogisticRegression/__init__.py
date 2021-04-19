# -*- coding: utf-8 -*-

import os

from json import encoder
from json import dumps

from sklearn_export.estimator.classifier.Classifier import Classifier


class LogisticRegression(Classifier):

    # @formatter:on
    def __init__(self, estimator, **kwargs):
        """
        Port a trained estimator to a dict.

        Parameters
        ----------
        :param estimator : LogisticRegression
            An instance of a trained LogisticRegression estimator.
        """
        super(LogisticRegression, self).__init__(
            estimator, **kwargs)

        self.estimator = estimator

    def load_model_data(self, model_data=None):

        if model_data is None:
            model_data = {}

        if 'type' not in model_data:
            model_data['type'] = ''

        model_data['coefficients'] = self.estimator.coef_.flatten('F').tolist()
        model_data['numRows'] = self.estimator.coef_.shape[0]
        model_data['numColumns'] = self.estimator.coef_.shape[1]
        model_data['intercept'] = self.estimator.intercept_.tolist()

        if self.estimator.multi_class is 'multinomial':
            if len(self.estimator.classes_) > 2:
                model_data['type'] += 'MultinomialLogisticRegression'
            else:
                model_data['type'] += 'BinaryLogisticRegression'
        elif len(self.estimator.classes_) > 2:
            model_data['type'] += 'MulticlassLogisticRegression'
        else:
            model_data['type'] += 'BinaryLogisticRegression'

        return model_data