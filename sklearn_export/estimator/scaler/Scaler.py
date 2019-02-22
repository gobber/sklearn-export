# -*- coding: utf-8 -*-

from sklearn_export.Template import Template


class Scaler(Template):

    def __init__(self, estimator, **kwargs):
        # pylint: disable=unused-argument
        super(Scaler, self).__init__(estimator, **kwargs)
        self.estimator_type = 'scaler'
