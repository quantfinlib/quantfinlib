

The design of QuantFinLib heavily uses concepts from Scikit-Learn. We aim to be compatible.


# Concepts


## Data Types

Generally, QuantFinLib works on any numeric data stored as numpy arrays or other types that are convertible to numeric arrays such as pandas DataFrame.
When possible we return the same data type from algorithms as the datatype they consume. For Pandas DataFrames we implement this with `set_output`.

https://scikit-learn.org/stable/developers/develop.html#developer-api-for-set-output

When we have time-series data in a Pandas DataFrame we aim to ensure that all transformation are backward looking, that the
transformed data at time t=T only depends on input data with t<=T. Sometimes this is however not the case, e.g. when we are creating
labels that are based on future values for supervised learning.


## Estimators

Estimators are the classes that learn and estimate some parameters of the data with the `fit()` method.

~~~
estimator = estimator.fit(data, targets) #supervised learning
sor
estimator = estimator.fit(data) #unsupervised learning
~~~

Some estimators in QuantFinLib are:

* Simulation models like `BrownianMotion`, `GeometricBrownianMotion`, `Garch`, `MarkovModel` who's simulation 
  parameters can be calibrated to historical data.


All estimators should inherit from `sklearn.base.BaseEstimator`.

## Transformers

Transformers are estimators that can transform data with `transform()` or `fit_transform()` methods.

~~~
new_data = transformer.transform(data)
new_data = transformer.fit_transform(data) 
~~~

Some transformers in QuantFinLib are:

* Indicator functions like `MovingAverage`.
* Simulation function that transform noise into price paths.

## Model

A model that can give a goodness of fit measure or a likelihood of unseen data, implements (higher is better):

~~~
score = model.score(data)
~~~

# Modules

## Simulation
## Statistics
## Indicators
