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
        Port a trained estimator to the syntax of a chosen programming
        language.

        Parameters
        ----------
        :param estimator : NuSVC
            An instance of a trained NuSVC estimator.
        :param target_language : string, default: 'java'
            The target programming language.
        :param target_method : string, default: 'predict'
            The target method of the estimator.
        """
        super(NuSVC, self).__init__(estimator, **kwargs)
