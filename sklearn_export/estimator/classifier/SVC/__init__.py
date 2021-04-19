# -*- coding: utf-8 -*-

from sklearn_export.estimator.classifier.Classifier import Classifier


class SVC(Classifier):
    """
    See also
    --------
    sklearn.svm.SVC

    http://scikit-learn.org/stable/modules/generated/
    sklearn.svm.SVC.html
    """

    # @formatter:on

    def __init__(self, estimator, **kwargs):
        """
        Port a trained estimator to a dict.

        Parameters
        ----------
        :param estimator : SVC
            An instance of a trained SVC estimator.
        """
        super(SVC, self).__init__(estimator, **kwargs)
        self.estimator = estimator

        est = self.estimator

        self.n_features = len(est.support_vectors_[0])
        self.svs_rows = est.n_support_
        self.n_svs_rows = len(est.n_support_)
        self.n_classes = len(est.classes_)
        self.params = est.get_params()

        # Kernel:
        self.kernel = str(self.params['kernel'])

        # Gamma:
        self.gamma = self.params['gamma']
        if self.gamma == 'auto':
            self.gamma = 1. / self.n_features
        self.gamma = self.repr(self.gamma)

        # Coefficient and degree:
        self.coef0 = self.repr(self.params['coef0'])
        self.degree = self.repr(self.params['degree'])

    def load_model_data(self, model_data=None):

        if model_data is None:
            model_data = {}

        if 'type' not in model_data:
            model_data['type'] = ''

        vectors = self.estimator.support_vectors_.flatten('F')
        coefficients = self.estimator.dual_coef_.flatten('F')

        model_data['vectors'] = vectors.tolist()
        model_data['coefficients'] = coefficients.tolist()
        model_data['intercepts'] = self.estimator.intercept_.tolist()
        model_data['weights'] = self.estimator.n_support_.tolist()
        model_data['kernel'] = self.kernel
        model_data['gamma'] = float(self.gamma)
        model_data['coef0'] = float(self.coef0)
        model_data['degree'] = float(self.degree)
        model_data['nClasses'] = int(self.n_classes)
        model_data['nRows'] = int(self.n_svs_rows)
        model_data['type'] += 'SVCBinary' if int(self.n_classes) == 2 else 'SVCMulticlass'
        model_data['numRowsV'] = self.estimator.support_vectors_.shape[0]
        model_data['numColumnsV'] = self.estimator.support_vectors_.shape[1]
        model_data['numRowsC'] = self.estimator.dual_coef_.shape[0]
        model_data['numColumnsC'] = self.estimator.dual_coef_.shape[1]

        return model_data