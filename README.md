# Classic Experiments

Persistent and reproducible experimental pipelines for Machine Learning.

## Installation

```
pip install git+git://github.com/jangop/classicexperiments
```

## Example Usage
We want to compare several classifiers with respect to a number of datasets.
We simply load the datasets and define a number of `Estimator` instances.
```python
import sklearn.ensemble
import sklearn.neighbors
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.tree
from classicdata import Ionosphere, MagicGammaTelescope

from classicexperiments import Estimator, Evaluation, Experiment

# Prepare datasets.
datasets = [Ionosphere(), MagicGammaTelescope()]

# Prepare estimators.
estimators = (
    [
        Estimator(
            name=f"{k}-nn",
            estimator_class=sklearn.neighbors.KNeighborsClassifier,
            parameters={"n_neighbors": k},
        )
        for k in (15, 23)
    ]
    + [
        Estimator(
            name="Tree",
            estimator_class=sklearn.tree.DecisionTreeClassifier,
            parameters={},
        )
    ]
    + [
        Estimator(
            name="Forest",
            estimator_class=sklearn.ensemble.AdaBoostClassifier,
            parameters={},
        )
    ]
    + [
        Estimator(
            name="MLP",
            estimator_class=sklearn.neural_network.MLPClassifier,
            parameters={},
        )
    ]
    + [
        Estimator(
            name="KernelSVM",
            estimator_class=sklearn.svm.SVC,
            parameters={"kernel": "sigmoid"},
        )
    ]
)

# Prepare experiments.
experiments = [
    Experiment(
        dataset=dataset,
        estimator=estimator,
        estimation_function=sklearn.model_selection.cross_val_score,
        parameters={},
        scaler=sklearn.preprocessing.StandardScaler(),
    )
    for estimator in estimators
    for dataset in datasets
]

# Prepare evaluation.
evaluation = Evaluation(experiments=experiments, base_dir="evaluation")

# Run evaluation.
evaluation.run()

# Present results.
evaluation.present(table_format="github")
```
Results are automatically stored, and we end up with a tidy table.

| Dataset    | 15-nn        | 23-nn        | Forest       | KernelSVM    | MLP          | Tree         |
|------------|--------------|--------------|--------------|--------------|--------------|--------------|
| Ionosphere | 0.81 ±0.0295 | 0.78 ±0.0349 | 0.91 ±0.0549 | 0.95 ±0.0321 | 0.90 ±0.0405 | 0.86 ±0.0491 |
| Telescope  | 0.84 ±0.0051 | 0.84 ±0.0056 | 0.84 ±0.0050 | 0.87 ±0.0052 | 0.87 ±0.0050 | 0.82 ±0.0046 |
