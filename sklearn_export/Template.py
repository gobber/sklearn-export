# -*- coding: utf-8 -*-

import os

from json import encoder
from json import dumps

class Template(object):

    def __init__(self, estimator, **kwargs):
        # pylint: disable=unused-argument
        # Default settings:
        self.num_format = lambda x: str(x)
        self.use_file = False

    def repr(self, value):
        return self.num_format(value)

    def to_json(self, directory, filename, model_data=None, with_md5_hash=False):
        """
        Save model data in a JSON file.

        Parameters
        ----------
        :param directory : string
            The directory.
        :param filename : string
            The filename.
        :param model_data: dict, default: None
            The output data in dict format.
        :param with_md5_hash : bool, default: False
            Whether to append the checksum to the filename or not.
        """
        model_data = self.load_model_data(model_data=model_data)

        encoder.FLOAT_REPR = lambda o: self.repr(o)
        json_data = dumps(model_data, sort_keys=True)
        if with_md5_hash:
            import hashlib
            json_hash = hashlib.md5(json_data).hexdigest()
            filename = filename.split('.json')[0] + '_' + json_hash + '.json'
        path = os.path.join(directory, filename)
        with open(path, 'w') as fp:
            fp.write(json_data)
        return model_data
