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
