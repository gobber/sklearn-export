# -*- coding: utf-8 -*-

from sklearn_export.estimator.classifier.SVC import SVC


class NuSVC(SVC):
    """
    See also
    --------
    sklearn.svm.NuSVC

    http://scikit-learn.org/stable/modules/generated/
    sklearn.svm.NuSVC.html
    """
    def __init__(self, estimator, **kwargs):
        """
        Port a trained estimator to a dict.

        Parameters
        ----------
        :param estimator : NuSVC
            An instance of a trained NuSVC estimator.
        """
        super(NuSVC, self).__init__(estimator, **kwargs)
