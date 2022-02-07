"""
Experiments and evaluations.
"""
import inspect
import operator
import os
import string
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Type

import numpy as np
import sklearn.pipeline
from classicdata.dataset import Dataset
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class ReproducibilityWarning(UserWarning):
    """
    Reminder to fix seeds.
    """


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

        self.check_reproducibility()

    @property
    def estimator_instance(self) -> sklearn.base.BaseEstimator:
        """
        An instance of this estimator.
        """
        if self._estimator_instance is None:
            self._estimator_instance = self.estimator_class(**self.parameters)
        return self._estimator_instance

    def check_reproducibility(self):
        """
        Emits a warning if the available parameters and the parameters actually set indicate
        that the estimator makes use of a random seed, without it being defined.
        """
        signature = inspect.signature(self.estimator_class)
        rnd_indicators = ["random_state", "seed", "random_seed"]
        for indicator in rnd_indicators:
            if indicator in signature.parameters and indicator not in self.parameters:
                warnings.warn(
                    f"{self.name} {self.estimator_class} has {indicator} amongst its parameters, "
                    f"which indicates random number generation. However, no value for {indicator} "
                    f"is passed. Results will probably not be reproducible.",
                    ReproducibilityWarning,
                )


class Experiment:
    """
    An experiment where one estimator is trained on one dataset.
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        dataset: Dataset,
        estimator: Estimator,
        estimation_function: Callable,
        parameters: dict,
        scaler: Optional[sklearn.base.TransformerMixin] = None,
    ):
        self.dataset = dataset
        self.estimator = estimator
        self.estimation_function = estimation_function
        self.parameters = parameters
        self.scaler = scaler

        self._name = None
        self.results = None

    @property
    def name(self):
        """
        Name of the experiment.
        """
        if self._name is None:
            self._name = self._construct_name()
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    def _construct_name(self):
        name = f"{self.dataset.short_name} {self.estimator.name} {self.estimator.parameters}"
        return name

    def run(self):
        """
        Train the estimator on the dataset, and store the results.
        :return:
        """
        if self.scaler is not None:
            pipeline = sklearn.pipeline.make_pipeline(
                self.scaler, self.estimator.estimator_instance
            )
        else:
            pipeline = self.estimator.estimator_instance

        self.results = self.estimation_function(
            pipeline, self.dataset.points, self.dataset.labels, **self.parameters
        )

    def save(self, path):
        """
        Store results.
        :param path: Where results are stored.
        """
        np.save(path, self.results)

    def load(self, path):
        """
        Load results.
        :param path: Where results are stored.
        """
        self.results = np.load(path)


class Evaluation:
    """
    Evaluate a number of experiments.
    """

    def __init__(self, experiments, base_dir):
        self._experiments = experiments
        self._base_dir = base_dir
        self._missing = None

    @property
    def datasets(self) -> set[Dataset]:
        """
        Every dataset that appears in at least one experiment.
        """
        return {experiment.dataset for experiment in self._experiments}

    def _prepare(self):
        """
        Collect those experiments for which results still need to be computed.
        """
        self._missing = []
        for experiment in self._experiments:
            experiment_path = os.path.join(self._base_dir, simplify(experiment.name))
            result_path = os.path.join(experiment_path, "result.npy")
            try:
                experiment.load(result_path)
            except FileNotFoundError:
                self._missing.append(experiment)

    def run(self):
        """
        Compute results for experiments that still need them.
        """
        if self._missing is None:
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

    def present(self, table_format: str = "simple"):
        """
        Present results.
        """

        @dataclass(order=True)
        class ReducedResult:
            """
            Reduces results; merely the mean and standard variance.
            """

            mean: float
            std: float
            highlight: bool = False

            @classmethod
            def from_results(cls, results: np.ndarray) -> "ReducedResult":
                """
                Reduces results by computing the mean and standard variance.
                """
                return ReducedResult(np.mean(results).item(), np.std(results).item())

            def __str__(self) -> str:
                representation = f"{self.mean:.2f} ±{self.std:.4f}"
                if self.highlight:
                    representation = f"**{representation}**"
                return representation

        def highlight(row: dict[ReducedResult]):
            """
            Determines and highlights the highest values.
            Modifies in place.

            :param row: The dictionary whose highest value(s) will be highlighted.
            :return: The dictionary with its highest value(s) highlighted.
            """
            best = max(row.values())
            for value in row.values():
                if value == best:
                    value.highlight = True
            return row

        data = [
            {"Dataset": dataset.short_name}
            | highlight(
                {
                    experiment.estimator.name: ReducedResult.from_results(
                        experiment.results
                    )
                    for experiment in self._experiments
                    if experiment.dataset is dataset
                }
            )
            for dataset in sorted(
                list(self.datasets), key=operator.attrgetter("short_name")
            )
        ]

        print(tabulate(data, headers="keys", tablefmt=table_format))


def keep(original: str, allowed: str = string.ascii_lowercase + string.digits) -> str:
    """
    Discard characters from strings.

    :param original: String from which characters get removed.
    :param allowed: String with all characters that are kept.
    :return: String without discarded characters.
    """
    new = ""
    for character in original:
        if character in allowed:
            new += character
    return new


def simplify(original: str) -> str:
    """
    Simplifies a string by replacing uppercase characters and whitespace,
    and removing non-ascii characters.
    :param original: Original string.
    :return: Simplified string.
    """
    simple = "-".join([keep(piece) for piece in original.lower().split()])
    return simple
