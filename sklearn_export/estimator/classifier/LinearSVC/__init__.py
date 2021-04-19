# -*- coding: utf-8 -*-

from sklearn_export.estimator.classifier.Classifier import Classifier


class LinearSVC(Classifier):
    """
    See also
    --------
    sklearn.svm.LinearSVC

    http://scikit-learn.org/stable/modules/generated/
    sklearn.svm.LinearSVC.html
    """

    # @formatter:on

    def __init__(self, estimator, **kwargs):
        """
        Port a trained estimator to a dict.

        Parameters
        ----------
        :param estimator : LinearSVC
            An instance of a trained LinearSVC estimator.
        """
        super(LinearSVC, self).__init__(estimator, **kwargs)
        self.estimator = estimator

        est = self.estimator

        self.n_features = len(est.coef_[0])
        self.n_classes = len(est.classes_)
        self.is_binary = est.n_classes == 2
        self.prefix = 'binary' if self.is_binary else 'multi'

    def load_model_data(self, model_data=None):

        if model_data is None:
            model_data = {}

        if 'type' not in model_data:
            model_data['type'] = ''

        est = self.estimator
        coeffs = est.coef_[0] if self.is_binary else est.coef_.flatten('F')
        inters = est.intercept_

        model_data['coefficients'] = coeffs.tolist()
        model_data['intercepts'] = inters.tolist()
        model_data['type'] += 'LinearSVCBinary' if self.is_binary else 'LinearSVCMulticlass'

        if self.is_binary:
            model_data['numRowsC'] = 1
            model_data['numColumnsC'] = est.coef_.shape[0]
        else:
            model_data['numRowsC'] = est.coef_.shape[0]
            model_data['numColumnsC'] = est.coef_.shape[1]

        return model_data
