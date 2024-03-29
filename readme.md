# sklearn-export

This package is based on sklearn porter from [https://github.com/nok/sklearn-porter](https://github.com/nok/sklearn-porter). We choose to build it because sklearn porter saves data in matrix format. However, many popular algebra libraries (e.g., [blas](http://www.netlib.org/blas/) and [lapack](http://www.netlib.org/lapack/)) are used to work with vectors. Then, sklearn-export saves the sklearn model data in Json format (matrices are stored in [column major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order)).  Note that, this is a beta version yet, then only some models and functionalities are supported.

## New features (0.0.7)

The code was optimized and now it works with sklearn >= 0.24. Some complete examples were added (see Complete Examples section).

## Support

|  Class | Details  |
| ------------ | ------ |
| [sklearn.svm.SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)| C-Support Vector Classification. The multiclass support is handled according to a one-vs-one scheme.|
| [sklearn.svm.NuSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html) | Nu-Support Vector Classification. Similar to SVC but uses a parameter to control the number of support vectors. |
|[sklearn.svc.LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) | Linear Support Vector Classification.|
|[sklearn.neural_network.MLPClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)| Multi-layer Perceptron classifier.|
|[sklearn.neural_network.MLPRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)|Multi-layer Perceptron regressor.|
|[sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)|Logistic Regression (aka logit, MaxEnt) classifier.|
|[sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)|Ordinary least squares Linear Regression.|
|[sklearn.preprocessing.MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)|Transforms features by scaling each feature to a given range.|
|[sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)|Standardize features by removing the mean and scaling to unit variance.|
|[sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)|Epsilon-Support Vector Regression.|
|[sklearn.svm.LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html)|Linear Support Vector Regression.|

**Observation**: details were extracted from sklearn documentation.
## Installation
We recommend to make a instalation using pip:
```bash
$ pip install sklearn_export
```
If you are using jupyter notebooks consider to install sklearn_export through a notebook cell. Then, you can type and execute the following:
```python
import sys
!{sys.executable} -m pip install sklearn_export
```
## Usage

Actually sklearn-export can save Classifiers, Regressions and some Scalers (see Support session).

### Saving a model or scaler

 The basic usage is to save a simple model.
```python
# Basic imports
from sklearn.datasets import load_iris
from sklearn_export import Export
from sklearn.neural_network import MLPRegressor

# Load data and train model
samples = load_iris()
X, y = samples.data, samples.target
clf = MLPRegressor()
clf.fit(X, y)

# Save using sklearn_export
export = Export(clf)
result = export.to_json()
```
The result is a JSON file that can be loaded in any programing language that supports JSON.

### Complete examples

Some complete examples are provided [here](https://github.com/gobber/sklearn-export/tree/master/examples).

### Saving multiple models
The sklearn-export can also save more then one model in the same JSON file. This is usefull when you need to store a Classifier and a Scaler (for example). Currently, it is only possible to export a Model and a Scaler, but future upgrades will include multiple models.
```python
# Basic imports
from sklearn.datasets import load_iris
from sklearn_export import Export
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Load data
samples = load_iris()
X, y = samples.data, samples.target

# Normalize data
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

# Train model with normalized data
clf = MLPRegressor()
clf.fit(Xz, y)

# Save a model and a scaler using sklearn_export
export = Export([scaler, clf])
result = export.to_json()
```
 The result is a Json file that contains information about a Model and a Scaler. The file can be loaded in any programing language that supports JSON files.

### Extra options

The method `to_json()` also support some other parameters:

|  Parameter | Details  | Default |
| -------- | ------ | ------ |
| `filename` | Name of the output Json file | `data.json` |
| `directory` | Path to save the file | `.` |
| `with_md5_hash` | Include md5 hash in file name | `False` |

## Questions
If you have any question please send me an email <charles26f@gmail.com>.

