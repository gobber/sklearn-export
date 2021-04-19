# -*- coding: utf-8 -*-

from sklearn_export.estimator.scaler.Scaler import Scaler


class MinMaxScaler(Scaler):

    def __init__(self, estimator, **kwargs):
        """
        Port a trained estimator to a dict.

        Parameters
        ----------
        :param estimator : LinearRegression
            An instance of a trained LinearRegression estimator.
        """
        super(MinMaxScaler, self).__init__(estimator,  **kwargs)

        self.estimator = estimator
        self.params = estimator.get_params()

    def load_model_data(self, model_data=None):

        if model_data is None:
            model_data = {}                    

        model_data['lower'] = self.params['feature_range'][0]
        model_data['upper'] = self.params['feature_range'][1]
        model_data['min'] = self.estimator.data_min_.tolist()
        model_data['max'] = self.estimator.data_max_.tolist()
        model_data['scaler'] = 'MinMaxScaler'

        return model_data