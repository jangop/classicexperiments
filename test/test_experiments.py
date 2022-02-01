import unittest

import sklearn.model_selection
import sklearn.neighbors
from classicdata import Ionosphere, MagicGammaTelescope

from classicexperiments import Estimator, Evaluation, Experiment


class TrainingTest(unittest.TestCase):
    def test(self):
        # Prepare datasets.
        datasets = [Ionosphere(), MagicGammaTelescope()]

        # Prepare estimators.
        knn3 = Estimator(
            name="3-nn",
            estimator_class=sklearn.neighbors.KNeighborsClassifier,
            parameters={"n_neighbors": 3},
        )
        knn5 = Estimator(
            name="5-nn",
            estimator_class=sklearn.neighbors.KNeighborsClassifier,
            parameters={"n_neighbors": 5},
        )
        estimators = [knn3, knn5]

        # Prepare experiments.
        experiments = [
            Experiment(
                dataset=dataset,
                estimator=estimator,
                estimation_function=sklearn.model_selection.cross_val_score,
                parameters={},
                scaler=None,
            )
            for estimator in estimators
            for dataset in datasets
        ]

        # Prepare evaluation.
        evaluation = Evaluation(experiments=experiments, base_dir="evaluation")

        # Run evaluation.
        evaluation.run()

        # Present results.
        evaluation.present()
