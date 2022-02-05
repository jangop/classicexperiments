# Classic Experiments

Persistent and reproducible experimental pipelines for Machine Learning.

## Installation

```
pip install classicexperiments
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
from classicdata import (
    USPS,
    ImageSegmentation,
    Ionosphere,
    LetterRecognition,
    MagicGammaTelescope,
    PenDigits,
    RobotNavigation,
)

from classicexperiments import Estimator, Evaluation, Experiment

# Prepare datasets.
datasets = [
    Ionosphere(),
    LetterRecognition(),
    MagicGammaTelescope(),
    PenDigits(),
    RobotNavigation(),
    ImageSegmentation(),
    USPS(),
]

# Prepare estimators.
estimators = [
    Estimator(
        name="Dummy",
        estimator_class=sklearn.dummy.DummyClassifier,
        parameters={},
    ),
    Estimator(
        name="5-nn",
        estimator_class=sklearn.neighbors.KNeighborsClassifier,
        parameters={"n_neighbors": 5},
    ),
    Estimator(
        name="Tree",
        estimator_class=sklearn.tree.DecisionTreeClassifier,
        parameters={},
    ),
    Estimator(
        name="Forest",
        estimator_class=sklearn.ensemble.AdaBoostClassifier,
        parameters={},
    ),
    Estimator(
        name="MLP",
        estimator_class=sklearn.neural_network.MLPClassifier,
        parameters={},
    ),
    Estimator(
        name="KernelSVM",
        estimator_class=sklearn.svm.SVC,
        parameters={"kernel": "sigmoid"},
    ),
]

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

| Dataset            | Dummy        | 5-nn         | Tree             | Forest           | MLP              | KernelSVM    |
|--------------------|--------------|--------------|------------------|------------------|------------------|--------------|
| Ionosphere         | 0.64 ±0.0036 | 0.83 ±0.0388 | 0.86 ±0.0491     | **0.91 ±0.0549** | 0.90 ±0.0405     | 0.84 ±0.0630 |
| Letter Recognition | 0.04 ±0.0001 | 0.94 ±0.0022 | 0.88 ±0.0051     | 0.26 ±0.0356     | **0.95 ±0.0044** | 0.47 ±0.0119 |
| Pen Digits         | 0.10 ±0.0000 | 0.99 ±0.0022 | 0.96 ±0.0048     | 0.43 ±0.1198     | **0.99 ±0.0017** | 0.74 ±0.0067 |
| Robot Navigation   | 0.40 ±0.0001 | 0.77 ±0.0563 | **0.98 ±0.0140** | 0.80 ±0.0365     | 0.87 ±0.0472     | 0.48 ±0.0272 |
| Segmentation       | 0.14 ±0.0000 | 0.92 ±0.0505 | 0.94 ±0.0334     | 0.48 ±0.0700     | **0.95 ±0.0362** | 0.75 ±0.0914 |
| Telescope          | 0.65 ±0.0001 | 0.81 ±0.0060 | 0.82 ±0.0046     | 0.84 ±0.0050     | **0.87 ±0.0050** | 0.65 ±0.0043 |
| USPS               | 0.17 ±0.0003 | 0.96 ±0.0030 | 0.88 ±0.0088     | 0.55 ±0.0898     | **0.97 ±0.0049** | 0.88 ±0.0053 |
