"""
Experiments and evaluations.
"""

import operator
import os
import string
from typing import Callable, Optional, Type

import more_termcolor
import numpy as np
import sklearn.pipeline
from classicdata.dataset import Dataset
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class Estimator:
    """
    Wrapper for an estimator from scikit-learn.

    Besides the class it holds a (pretty) name and parameters used for instantiation.
    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        name: str,
        estimator_class: Type[sklearn.base.BaseEstimator],
        parameters: dict,
    ):
        self.name = name
        self.estimator_class = estimator_class
        self.parameters = parameters

        self._estimator_instance = None

    @property
    def estimator_instance(self) -> sklearn.base.BaseEstimator:
        """
        An instance of this estimator.
        """
        if self._estimator_instance is None:
            self._estimator_instance = self.estimator_class(**self.parameters)
        return self._estimator_instance


class Experiment:
    """
    An experiment where one estimator is trained on one dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        estimator: Estimator,
        estimation_function: Callable,
        parameters: dict,
        scaler: Optional[sklearn.base.TransformerMixin] = None,
    ):
        self._dataset = dataset
        self._estimator = estimator
        self._estimation_function = estimation_function
        self._parameters = parameters
        self._scaler = scaler

    @property
    def name(self):
        try:
            self._name
        except AttributeError:
            self._name = self._construct_name()
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def estimator(self):
        return self._estimator

    @property
    def dataset(self):
        return self._dataset

    @property
    def results(self):
        return self._results

    def _construct_name(self):
        name = "{dataset} {estimator} {parameters}".format(
            dataset=self._dataset.short_name,
            estimator=self._estimator.name,
            parameters=self._estimator.parameters,
        )
        return name

    def run(self):
        if self._scaler is not None:
            pipeline = sklearn.pipeline.make_pipeline(
                self._scaler, self._estimator.estimator_instance
            )
        else:
            pipeline = self._estimator.estimator_instance

        self._results = self._estimation_function(
            pipeline, self._dataset.points, self._dataset.labels, **self._parameters
        )

    def save(self, path):
        np.save(path, self._results)

    def load(self, path):
        self._results = np.load(path)


class Evaluation:
    def __init__(self, experiments, base_dir):
        self._experiments = experiments
        self._base_dir = base_dir

    def _prepare(self):
        self._missing = []
        for experiment in self._experiments:
            experiment_path = os.path.join(self._base_dir, simplify(experiment.name))
            result_path = os.path.join(experiment_path, "result.npy")
            try:
                experiment.load(result_path)
            except FileNotFoundError:
                self._missing.append(experiment)

    def run(self):
        try:
            self._missing
        except AttributeError:
            self._prepare()
        if self._missing:
            missing_datasets = list(
                {experiment.dataset for experiment in self._missing}
            )
            missing_per_dataset = {
                dataset.short_name: [
                    experiment.estimator.name
                    for experiment in self._missing
                    if experiment.dataset == dataset
                ]
                for dataset in missing_datasets
            }

            for dataset_name, estimator_names in missing_per_dataset.items():
                logger.info(
                    f'{dataset_name} lacks results for {", ".join(estimator_names)}'
                )

        for experiment in tqdm(self._missing, desc="Running experiments"):
            logger.info(
                f"Running {experiment.dataset.short_name}: {experiment.estimator.name}..."
            )
            experiment.run()
            experiment_path = os.path.join(self._base_dir, simplify(experiment.name))
            result_path = os.path.join(experiment_path, "result.npy")
            os.makedirs(experiment_path, exist_ok=True)
            experiment.save(result_path)

    def present(self):
        mode = "plain-table"
        if mode == "plain-list":
            for experiment in self._experiments:
                logger.info(
                    "Mean cross validation score for {} on {}: {:.2f} ±{:.4f}".format(
                        experiment.estimator.name,
                        experiment.dataset.short_name,
                        np.mean(experiment.results),
                        np.std(experiment.results),
                    )
                )
        elif mode == "plain-table":
            table = []
            datasets = sorted(
                list({experiment.dataset for experiment in self._experiments}),
                key=operator.attrgetter("short_name"),
            )
            estimators = sorted(
                list({experiment.estimator for experiment in self._experiments}),
                key=operator.attrgetter("name"),
            )
            for estimator in estimators:
                estimator.name = estimator.name.replace("Early Stopping", "ES")
                estimator.name = estimator.name.replace("Reg", "R")
                estimator.name = estimator.name.replace("LR", "L")
                estimator.name = estimator.name.replace(" ", "")

            header = ["Dataset"] + [estimator.name for estimator in estimators]
            for dataset in datasets:
                row = [dataset.short_name]
                best_cols = []
                best_mean = -1
                for i_col, estimator in enumerate(estimators):
                    experiment = [
                        experiment
                        for experiment in self._experiments
                        if experiment.dataset is dataset
                        and experiment.estimator is estimator
                    ][0]
                    try:
                        mean = np.mean(experiment.results)
                        std = np.std(experiment.results)

                    except AttributeError:
                        value = "nan"
                    else:
                        value = f"{mean:.2f} ±{std:.4f}"

                        if mean > best_mean:
                            best_mean = mean
                            best_cols = [i_col]
                        elif mean == best_mean:
                            best_cols.append(i_col)

                    row += [value]

                for i_col in best_cols:
                    i_col += 1
                    row[i_col] = more_termcolor.colors.green(row[i_col])

                table.append(row)
            print(tabulate(table, headers=header))


def keep(original, allowed=string.ascii_lowercase + string.digits):
    new = ""
    for c in original:
        if c in allowed:
            new += c
    return new


def simplify(original):
    simple = "-".join([keep(piece) for piece in original.lower().split()])
    return simple
