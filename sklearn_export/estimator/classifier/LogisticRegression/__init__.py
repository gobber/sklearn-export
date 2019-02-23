# -*- coding: utf-8 -*-

import os

from json import encoder
from json import dumps

from sklearn_export.estimator.classifier.Classifier import Classifier


class LogisticRegression(Classifier):

    # @formatter:on
    def __init__(self, estimator, **kwargs):

        super(LogisticRegression, self).__init__(
            estimator, **kwargs)

        self.estimator = estimator

    def load_model_data(self, model_data=None):

        if model_data is None:
            model_data = {}

        if 'type' not in model_data:
            model_data['type'] = ''

        model_data['coefficients'] = self.estimator.coef_.flatten('F').tolist()
        model_data['numRows'] = self.estimator.coef_.shape[0]
        model_data['numColumns'] = self.estimator.coef_.shape[1]
        model_data['intercept'] = self.estimator.intercept_.tolist()

        if self.estimator.multi_class is 'multinomial':
            if len(self.estimator.classes_) > 2:
                model_data['type'] += 'MultinomialLogisticRegression'
            else:
                model_data['type'] += 'BinaryLogisticRegression'
        elif len(self.estimator.classes_) > 2:
            model_data['type'] += 'MulticlassLogisticRegression'
        else:
            model_data['type'] += 'BinaryLogisticRegression'

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
        :param with_md5_hash : bool, default: False
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
