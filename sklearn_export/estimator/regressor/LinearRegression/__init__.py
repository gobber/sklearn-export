# -*- coding: utf-8 -*-

import os

from json import encoder
from json import dumps

from sklearn_export.estimator.regressor.Regressor import Regressor


class LinearRegression(Regressor):
    """
    See also
    --------
    sklearn.linear_models.LinearRegression

    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """

    # @formatter:on
    def __init__(self, estimator, **kwargs):
        """
        Port a trained estimator to the syntax of a chosen programming
        language.

        Parameters
        ----------
        :param estimator : LinearRegression
            An instance of a trained LinearRegression estimator.
        :param target_language : string
            The target programming language.
        :param target_method : string
            The target method of the estimator.
        """
        super(LinearRegression, self).__init__(estimator, **kwargs)

        self.estimator = estimator

    def load_model_data(self, model_data=None):

        if model_data is None:
            model_data = {}

        if 'type' not in model_data:
            model_data['type'] = ''

        model_data['coefficients'] = self.estimator.coef_.tolist()
        model_data['intercept'] = [self.estimator.intercept_.tolist()]
        model_data['type'] += 'LinearRegression'

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
            json_hash = hashlib.md5(json_data).hexdigest()
            filename = filename.split('.json')[0] + '_' + json_hash + '.json'
        path = os.path.join(directory, filename)
        with open(path, 'w') as fp:
            fp.write(json_data)
