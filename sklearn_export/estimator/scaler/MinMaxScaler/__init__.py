# -*- coding: utf-8 -*-

import os

from json import encoder
from json import dumps

from sklearn_export.estimator.scaler.Scaler import Scaler


class MinMaxScaler(Scaler):

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
        super(MinMaxScaler, self).__init__(estimator,  **kwargs)

        self.estimator = estimator
        self.params = estimator.get_params()

    def load_model_data(self, model_data=None):

        if model_data is None:
            model_data = {}

        if 'type' not in model_data:
            model_data['type'] = ''

        model_data['lower'] = self.params['feature_range'][0]
        model_data['upper'] = self.params['feature_range'][1]
        model_data['min'] = self.estimator.data_min_.tolist()
        model_data['max'] = self.estimator.data_max_.tolist()
        model_data['type'] = 'MinMaxScaler'

        return model_data

    def to_json(self, directory, filename, model_data=None, with_md5_hash=False):

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

