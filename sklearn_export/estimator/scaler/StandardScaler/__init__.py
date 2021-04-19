# -*- coding: utf-8 -*-

from sklearn_export.estimator.scaler.Scaler import Scaler


class StandardScaler(Scaler):

    def __init__(self, estimator, **kwargs):
        """
        Port a trained estimator to a dict.

        Parameters
        ----------
        :param estimator : LinearRegression
            An instance of a trained LinearRegression estimator.
        """
        super(StandardScaler, self).__init__(estimator,  **kwargs)

        self.estimator = estimator
        self.params = estimator.get_params()

    def load_model_data(self, model_data=None):

        if model_data is None:
            model_data = {}

        if self.params['with_std'] and self.params['with_mean']:
            model_data['scaler'] = 'ZscoreScaler'
            model_data['mean'] = self.estimator.mean_.tolist()
            model_data['std'] = self.estimator.scale_.tolist()
        elif self.params['with_std']:
            model_data['scaler'] = 'StandardDeviationScaler'
            model_data['std'] = self.estimator.scale_.tolist()
        elif self.params["with_mean"]:
            model_data['scaler'] = 'MeanScaler'
            model_data['mean'] = self.estimator.mean_.tolist()
        else:
            raise AttributeError('You need mean or std to normalize.')

        return model_data

