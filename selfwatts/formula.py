# Copyright (C) 2018  INRIA
# Copyright (C) 2018  University of Lille
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import hashlib
import logging
import pickle
import warnings
from collections import OrderedDict, deque
from typing import List, Dict

from scipy import stats
from scipy.linalg import LinAlgWarning
from scipy.stats import PearsonRConstantInputWarning
from sklearn.linear_model import Lasso as Regression

from selfwatts.topology import CPUTopology

# make scikit-learn more silent
warnings.filterwarnings('ignore', category=LinAlgWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=PearsonRConstantInputWarning)


class PowerModelNotInitializedException(Exception):
    """
    This exception happens when a user try to compute a power estimation without having learned a power model.
    """
    pass


class NotEnoughReportsInHistoryException(Exception):
    """
    This exception happens when a user try to learn a power model without having enough reports in history.
    """
    pass


class History:
    """
    This class stores the reports history to use when learning a new power model.
    """

    def __init__(self, max_length) -> None:
        """
        Initialize a new reports history container.
        :param max_length: Maximum amount of samples to keep before overriding the oldest sample at insertion
        """
        self.max_length = max_length
        self.X: deque[List[int]] = deque(maxlen=max_length)
        self.y: deque[float] = deque(maxlen=max_length)
        self.sync = False

    def __len__(self) -> int:
        """
        Compute the length of the history.
        :return: Length of the history
        """
        return len(self.X)

    def store_report(self, power_reference: float, events_value: List[int]) -> None:
        """
        Append a report to the reports history.
        :param events_value: List of raw events value
        :param power_reference: Power reference corresponding to the events value
        """
        if len(events_value) > 0:
            self.X.append(events_value)
            self.y.append(power_reference)

    def generate_pearson_coef(self) -> List[float]:
        """
        Returns the pearson coefficient of the features.
        :return: List of pearson coefficient of the features.
        """
        correlations = []

        for feature, _ in enumerate(next(iter(self.X))):
            samples = []
            for sample in self.X:
                samples.append(sample[feature])

            correlations.append(stats.pearsonr(samples, self.y)[0])

        return correlations

    def flush(self) -> None:
        """
        Flush the samples from the history.
        """
        self.X.clear()
        self.y.clear()


class PowerModel:
    """
    This Power model compute the power estimations and handle the learning of a new model when needed.
    """

    def __init__(self, frequency: int, history_window_size: int) -> None:
        """
        Initialize a new power model.
        :param frequency: Frequency of the power model
        :param history_window_size: Size of the history window used to keep samples to learn from
        """
        self.frequency = frequency
        self.model: Regression = Regression()
        self.hash: str = 'uninitialized'
        self.history: History = History(history_window_size)
        self.id = 0

    def generate_unfitted_model(self) -> Regression:
        """
        Returns an unfitted model to use for the power estimations.
        :return: Unfitted regression model
        """
        # fit_intercept = True if len(self.history) == self.history.max_length else False
        return Regression()

    def old_learn_power_model(self, min_samples: int, min_intercept: float, max_intercept: float) -> None:
        """
        Learn a new power model using the stored reports and update the formula id/hash.
        :param min_samples: Minimum amount of samples required to learn the power model
        :param min_intercept: Minimum value allowed for the intercept of the model
        :param max_intercept: Maximum value allowed for the intercept of the model
        """
        if len(self.history) < min_samples:
            return

        try:
            model = self.generate_unfitted_model()
            model.fit(self.history.X, self.history.y)
        except ValueError:
            logging.error('failed to learn a new power model from history values')
            self.history.flush()
            return

        # Discard the new model when the intercept is not in specified range
        if not (min_intercept <= model.intercept_ < max_intercept):
            return

        self.model = model
        self.hash = hashlib.blake2b(pickle.dumps(self.model), digest_size=20).hexdigest()
        self.id += 1

    def learn_power_model(self, min_samples: int, min_intercept: float, max_intercept: float) -> None:
        """
        Learn a new power model using the stored reports and update the formula id/hash.
        :param min_samples: Minimum amount of samples required to learn the power model
        :param min_intercept: Minimum value allowed for the intercept of the model
        :param max_intercept: Maximum value allowed for the intercept of the model
        """
        if len(self.history) < min_samples:
            return

        try:
            model = self.generate_unfitted_model()
            model.fit(self.history.X, self.history.y)
        except ValueError:
            logging.error('failed to learn a new power model from history values')
            self.history.flush()
            return

        # Discard the new model when the intercept is not in specified range
        if not (min_intercept <= model.intercept_ < max_intercept):
            return

        self.model = model
        self.hash = hashlib.blake2b(pickle.dumps(self.model), digest_size=20).hexdigest()
        self.id += 1

    @staticmethod
    def _extract_events_value(events: Dict[str, int]) -> List[int]:
        """
        Creates and return a list of events value from the events group.
        :param events: Events group
        :return: List containing the events value sorted by event name
        """
        return [value for _, value in sorted(events.items())]

    def store_report_in_history(self, power_reference: float, events: Dict[str, int]) -> None:
        """
        Store the events group into the System reports list and learn a new power model.
        :param power_reference: Power reference (in Watt)
        :param events: Events value
        """
        self.history.store_report(power_reference, self._extract_events_value(events))

    def compute_power_estimation(self, events: Dict[str, int]) -> float:
        """
        Compute a power estimation from the events value using the power model.
        :param events: Events value
        :raise: NotFittedError when the model haven't been fitted
        :return: Power estimation for the given events value
        """
        return self.model.predict([self._extract_events_value(events)])[0]

    def cap_power_estimation(self, raw_target_power: float, raw_global_power: float) -> (float, float):
        """
        Cap target's power estimation to the global power estimation.
        :param raw_target_power: Target power estimation from the power model (in Watt)
        :param raw_global_power: Global power estimation from the power model (in Watt)
        :return: Capped power estimation (in Watt) with its ratio over global power consumption
        """
        target_power = raw_target_power - self.model.intercept_
        global_power = raw_global_power - self.model.intercept_

        ratio = target_power / global_power if global_power > 0.0 and target_power > 0.0 else 0.0
        power = target_power if target_power > 0.0 else 0.0
        return power, ratio

    def apply_intercept_share(self, target_power: float, target_ratio: float) -> float:
        """
        Apply the target's share of intercept from its ratio from the global power consumption.
        :param target_power: Target power estimation (in Watt)
        :param target_ratio: Target ratio over the global power consumption
        :return: Target power estimation including intercept (in Watt) and ratio over global power consumption
        """
        intercept = target_ratio * self.model.intercept_
        return target_power + intercept


class SelfWattsFormula:
    """
    This formula compute per-target power estimations using hardware performance counters.
    """

    def __init__(self, cpu_topology: CPUTopology, history_window_size: int) -> None:
        """
        Initialize a new formula.
        :param cpu_topology: CPU topology to use
        :param history_window_size: Size of the history window used to keep samples to learn from
        """
        self.cpu_topology = cpu_topology
        self.history_window_size = history_window_size
        self.models = self._gen_models_dict(history_window_size)

    def _gen_models_dict(self, history_window_size: int) -> Dict[int, PowerModel]:
        """
        Generate and returns a layered container to store per-frequency power models.
        :param history_window_size: Size of the history window used to keep samples to learn from
        :return: Initialized Ordered dict containing a power model for each frequency layer
        """
        return OrderedDict((freq, PowerModel(freq, history_window_size)) for freq in self.cpu_topology.get_supported_frequencies())

    def _get_frequency_layer(self, frequency: float) -> int:
        """
        Find and returns the nearest frequency layer for the given frequency.
        :param frequency: CPU frequency
        :return: Nearest frequency layer for the given frequency
        """
        last_layer_freq = 0
        for current_layer_freq in self.models.keys():
            if frequency < current_layer_freq:
                return last_layer_freq
            last_layer_freq = current_layer_freq

        return last_layer_freq

    def compute_pkg_frequency(self, system_msr: Dict[str, int]) -> float:
        """
        Compute the average package frequency.
        :param system_msr: MSR events group of System target
        :return: Average frequency of the Package
        """
        return (self.cpu_topology.get_base_frequency() * system_msr['APERF']) / system_msr['MPERF']

    def get_power_model(self, system_core: Dict[str, int]) -> PowerModel:
        """
        Fetch the suitable power model for the current frequency.
        :param system_core: Core events group of System target
        :return: Power model to use for the current frequency
        """
        return self.models[self._get_frequency_layer(self.compute_pkg_frequency(system_core))]
        # return next(iter(self.models.values()))

    def flush_power_models(self) -> None:
        """
        Remove all learned models and their samples history.
        """
        self.models = self._gen_models_dict(self.history_window_size)
