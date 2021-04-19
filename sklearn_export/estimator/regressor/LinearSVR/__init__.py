# -*- coding: utf-8 -*-

from sklearn_export.estimator.regressor.Regressor import Regressor


class LinearSVR(Regressor):
    """
        See also
        --------
        sklearn.svm.LinearSVR

        """

    # @formatter:on

    def __init__(self, estimator, **kwargs):
        """
        Port a trained estimator to a dict.

        Parameters
        ----------
        :param estimator : LinearSVR
            An instance of a trained LinearSVR estimator.
        """
        super(LinearSVR, self).__init__(estimator, **kwargs)
        self.estimator = estimator
        #self.is_binary = True if len(estimator.coef_.shape) == 1 else False

    def load_model_data(self, model_data=None):

        if model_data is None:
            model_data = {}

        if 'type' not in model_data:
            model_data['type'] = ''

        est = self.estimator
        coeffs = est.coef_
        inters = est.intercept_

        model_data['coefficients'] = coeffs.tolist()
        model_data['intercepts'] = inters.tolist()
        model_data['type'] += 'LinearSVR'

        #if self.is_binary:
        #    model_data['numRowsC'] = 1
        #    model_data['numColumnsC'] = est.coef_.shape[0]
        #else:
        #    model_data['numRowsC'] = est.coef_.shape[0]
        #    model_data['numColumnsC'] = est.coef_.shape[1]

        return model_data