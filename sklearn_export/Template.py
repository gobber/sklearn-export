# -*- coding: utf-8 -*-


class Template(object):

    def __init__(self, estimator, **kwargs):
        # pylint: disable=unused-argument
        # Default settings:
        self.num_format = lambda x: str(x)
        self.use_file = False

    def repr(self, value):
        return self.num_format(value)

    def data(self, dict_):
        copy = self.__dict__.copy()
        copy.update(dict_)  # update and extend dictionary
        return copy
