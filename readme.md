# sklearn-export

This package is based on sklearn port from [https://github.com/nok/sklearn-porter](https://github.com/nok/sklearn-porter).  I chose to build it because sklearn port saves data in matrix format. However, most popular algebra libraries are used to working with vectors. Then, sklearn-export saves the sklearn model data in Json format (as column vectors).  Note that this is a beta version yet, then only some models and functionalities are supported.

## New features (0.0.6)

The code was optimized and now it works with sklearn >= 0.24.

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
|[sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)|Standardize features by removing the mean and scaling to unit variance|
|[sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)|Epsilon-Support Vector Regression.|
|[sklearn.svm.LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html)|Linear Support Vector Regression.|

**Observation**: details where extracted from sklearn documentation.
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

 ### Saving a Model or Scaler

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
export.to_json()
```
The result is a Json file that can be load in any language.

### Saving a Model and a Scaler
The sklearn-export can also save more then one class in the same Json. This is usefull to store a Classifier and a Scaler (for example). To be honest, actually is only possible to store a pair Model and Scaler.
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

# Save model and scaler using sklearn_export
export = Export([scaler, clf])
export.to_json()
```
 The result is a Json file that contains information about a Model and a Scaler. The file can be load in any language.

### Extra options

The method `to_json()` also support some other parameters:

|  Parameter | Details  | Default |
| -------- | ------ | ------ |
| `filename` | Name of the output Json file | `data.json` |
| `directory` | Path to save the file | `.` |
| `with_md5_hash` | Include md5 hash in file name | `False` |

## Questions
If you have any question please send me a mail <charles26f@gmail.com>.

