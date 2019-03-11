# -*- coding: utf-8 -*-

import os

from json import encoder
from json import dumps

from sklearn_export.estimator.regressor.Regressor import Regressor


class LinearSVR(Regressor):
    """
        See also
        --------
        sklearn.svm.LinearSVR

        """

    # @formatter:on

    def __init__(self, estimator, **kwargs):

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

    def to_json(self, directory, filename, model_data=None, with_md5_hash=False):
        """
        Save model data in a JSON file.

        Parameters
        ----------
        :param directory : string
            The directory.
        :param filename : string
            The filename.
        :param with_md5_hash : bool
            Whether to append the checksum to the filename or not.
        """

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
