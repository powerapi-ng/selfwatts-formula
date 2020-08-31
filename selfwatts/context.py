import re
from enum import Enum
from typing import Dict

from powerapi.formula import FormulaState, FormulaActor
from powerapi.pusher import PusherActor

from selfwatts.topology import CPUTopology


class FormulaScope(Enum):
    """
    Enum used to set the scope of the SelfWatts formula.
    """
    CPU = "cpu"
    DRAM = "dram"


class SelfWattsFormulaConfig:
    """
    Global config of the SelfWatts formula.
    """

    def __init__(self, scope: FormulaScope, reports_frequency: int, rapl_event: str, error_threshold: float, cpu_topology: CPUTopology, min_samples_required: int, history_window_size: int):
        """
        Initialize a new formula config object.
        :param scope: Scope of the formula
        :param reports_frequency: Frequency at which the reports (in milliseconds)
        :param rapl_event: RAPL event to use as reference
        :param error_threshold: Error threshold (in Watt)
        :param cpu_topology: Topology of the CPU
        :param min_samples_required: Minimum amount of samples required before trying to learn a power model
        :param history_window_size: Size of the history window used to keep samples to learn from
        """
        self.scope = scope
        self.reports_frequency = reports_frequency
        self.rapl_event = rapl_event
        self.error_threshold = error_threshold
        self.cpu_topology = cpu_topology
        self.min_samples_required = min_samples_required
        self.history_window_size = history_window_size


class SelfWattsFormulaState(FormulaState):
    """
    State of the SelfWatts formula actor.
    """

    def __init__(self, actor: FormulaActor, pushers: Dict[str, PusherActor], config: SelfWattsFormulaConfig):
        """
        Initialize a new formula state object.
        :param actor: Actor of the formula
        :param pushers: Dictionary of available pushers
        :param config: Configuration of the formula
        """
        FormulaState.__init__(self, actor, pushers)
        self.config = config

        m = re.search(r'^\(\'(.*)\', \'(.*)\', \'(.*)\'\)$', actor.name)  # TODO: Need a better way to get these information
        self.dispatcher = m.group(1)
        self.sensor = m.group(2)
        self.socket = m.group(3)
