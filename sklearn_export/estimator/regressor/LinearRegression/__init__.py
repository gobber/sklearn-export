# -*- coding: utf-8 -*-

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
        Port a trained estimator to a dict.

        Parameters
        ----------
        :param estimator : LinearRegression
            An instance of a trained LinearRegression estimator.
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