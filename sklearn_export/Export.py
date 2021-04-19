
import sys
import types

#from sklearn.tree.tree import DecisionTreeClassifier
#from sklearn.ensemble.weight_boosting import AdaBoostClassifier
#from sklearn.ensemble.forest import RandomForestClassifier
#from sklearn.ensemble.forest import ExtraTreesClassifier
#from sklearn.neighbors.classification import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Export(object):

    def __init__(self, estimator, **kwargs):
        # pylint: disable=unused-argument
        """
        Transpile a trained estimator to the
        chosen target programming language.

        Parameters
        ----------

        """

        self.sklearn_ver = self.take_sklearn_version()

        if isinstance(estimator, list):

            if len(estimator) > 2:
                raise AttributeError('The length list must be at most 2.')

            self.template = []

            if isinstance(estimator[0], self._scalers) and isinstance(estimator[1], self._scalers):
                raise AttributeError('One of the estimators must be a Classifier or a Regressor model.')

            if isinstance(estimator[0], self._scalers):
                pass
            elif isinstance(estimator[1], self._scalers):
                temp = estimator[0]
                estimator[0] = estimator[1]
                estimator[1] = temp
            else:
                raise AttributeError('One of the estimators must be a Scaler.')

            self.template.append(self.load(estimator[0], **kwargs))
            del self.estimator
            self.template.append(self.load(estimator[1], **kwargs))
        else:
            self.template = self.load(estimator, **kwargs)

    @staticmethod
    def take_sklearn_version():
        from sklearn import __version__ as sklearn_ver
        sklearn_ver = str(sklearn_ver).split('.')
        sklearn_ver = [int(v) for v in sklearn_ver]
        major, minor = sklearn_ver[0], sklearn_ver[1]
        patch = sklearn_ver[2] if len(sklearn_ver) >= 3 else 0
        return major, minor, patch

    def load(self, estimator, **kwargs):

        # Extract estimator from 'Pipeline':
        # sklearn version >= 0.15.0
        #if not hasattr(self, 'estimator') and self.sklearn_ver[:2] >= (0, 15):
            #from sklearn.pipeline import Pipeline
            #if isinstance(estimator, Pipeline):
                #if hasattr(estimator, '_final_estimator') and \
                        #estimator._final_estimator is not None:
                    #self.estimator = estimator._final_estimator

        # Extract estimator from optimizer (GridSearchCV, RandomizedSearchCV):
        # sklearn version >= 0.19.0
        #if not hasattr(self, 'estimator') and self.sklearn_ver[:2] >= (0, 19):
            #from sklearn.model_selection._search import GridSearchCV
            #from sklearn.model_selection._search import RandomizedSearchCV
            #optimizers = (GridSearchCV, RandomizedSearchCV)
            #if isinstance(estimator, optimizers):
                #if hasattr(estimator, 'best_estimator_') and \
                        #hasattr(estimator.best_estimator_, '_final_estimator'):
                    #self.estimator = estimator.best_estimator_._final_estimator

        if not hasattr(self, 'estimator'):
            self.estimator = estimator

        # Determine the local supported estimators:
        self.supported_classifiers = self._classifiers
        self.supported_regressors = self._regressors
        self.supported_scalers = self._scalers

        # Read algorithm name and type:
        self.estimator_name = str(type(self.estimator).__name__)
        if isinstance(self.estimator, self.supported_classifiers):
            self.estimator_type = 'classifier'
        elif isinstance(self.estimator, self.supported_regressors):
            self.estimator_type = 'regressor'
        elif isinstance(self.estimator, self.supported_scalers):
            self.estimator_type = 'scaler'
        else:
            error = "Currently the given estimator '{estimator}' isn't" \
                    " supported.".format(**self.__dict__)
            raise ValueError(error)

        # Import estimator class:
        if sys.version_info[:2] < (3, 3):
            pckg = 'estimator.{estimator_type}.{estimator_name}'
            level = -1
        else:
            pckg = 'sklearn_export.estimator.{estimator_type}.{estimator_name}'
            level = 0
        pckg = pckg.format(**self.__dict__)
        try:
            clazz = __import__(pckg, globals(), locals(),
                               [self.estimator_name], level)
            clazz = getattr(clazz, self.estimator_name)
        except ImportError:
            error = "Currently the given model '{algorithm_name}' " \
                    "isn't supported.".format(**self.__dict__)
            raise AttributeError(error)

        # Create instance with all parameters:
        return clazz(**self.__dict__)

    def to_json(self, num_format=lambda x: str(x), directory='.', filename='data.json', with_md5_hash=False, **kwargs):

        if isinstance(num_format, types.LambdaType):
            if isinstance(self.template, list):
                self.template[0].num_format = num_format
                self.template[1].num_format = num_format
                return self.template[1].to_json(directory=directory,
                                         filename=filename,
                                         with_md5_hash=with_md5_hash,
                                         model_data=self.template[0].load_model_data(),
                                         **kwargs)
            else:
                self.template.num_format = num_format
                return self.template.to_json(directory=directory,
                                      filename=filename,
                                      with_md5_hash=with_md5_hash,
                                      **kwargs)

    @property
    def _classifiers(self):
        """
        Get a set of supported classifiers.

        Returns
        -------
        classifiers : {set}
            The set of supported classifiers.
        """

        # sklearn version < 0.18.0
        classifiers = (
            #AdaBoostClassifier,
            #BernoulliNB,
            #DecisionTreeClassifier,
            #ExtraTreesClassifier,
            #GaussianNB,
            #KNeighborsClassifier,
            LogisticRegression,
            #RandomForestClassifier,
        )

        if self.sklearn_ver[:2] < (0, 24):
            from sklearn.svm.classes import LinearSVC
            from sklearn.svm.classes import SVC
            from sklearn.svm.classes import NuSVC
        else:
            from sklearn.svm import LinearSVC
            from sklearn.svm import SVC
            from sklearn.svm import NuSVC

        classifiers += (LinearSVC, SVC, NuSVC)

        # sklearn version >= 0.18.0
        if self.sklearn_ver[:2] >= (0, 18):
            if self.sklearn_ver[:2] < (0, 24):
                from sklearn.neural_network.multilayer_perceptron import MLPClassifier
            else:
                from sklearn.neural_network import MLPClassifier
            classifiers += (MLPClassifier, )

        return classifiers

    @property
    def _regressors(self):
        """
        Get a set of supported regressors.

        Returns
        -------
        regressors : {set}
            The set of supported regressors.
        """

        # sklearn version < 0.18.0
        regressors = (
            LinearRegression,
        )

        if self.sklearn_ver[:2] < (0, 24):
            from sklearn.svm.classes import SVR, LinearSVR
        else:
            from sklearn.svm import SVR, LinearSVR
        regressors += (SVR, LinearSVR)

        # sklearn version >= 0.18.0
        if self.sklearn_ver[:2] >= (0, 18):
            if self.sklearn_ver[:2] < (0, 24):
                from sklearn.neural_network.multilayer_perceptron import MLPRegressor
            else:
                from sklearn.neural_network import MLPRegressor
            regressors += (MLPRegressor, )

        return regressors

    @property
    def _scalers(self):
        """
        Get a set of supported regressors.

        Returns
        -------
        scalers : {set}
            The set of supported regressors.
        """

        # sklearn version < 0.18.0
        scalers = (
            StandardScaler,
            MinMaxScaler,
        )

        return scalers
