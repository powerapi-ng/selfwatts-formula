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

from typing import List


class CPUTopology:
    """
    This class stores the necessary information about the CPU topology.
    """

    def __init__(self, tdp: int, freq_bclk: int, ratio_min: int, ratio_base: int, ratio_max: int, fixed_counters: int, general_counters: int) -> None:
        """
        Create a new CPU topology object.
        :param tdp: TDP of the CPU in Watt
        :param freq_bclk: Base clock in MHz
        :param ratio_min: Maximum efficiency ratio
        :param ratio_base: Base frequency ratio
        :param ratio_max: Maximum frequency ratio (with Turbo-Boost)
        :param fixed_counters: Number of fixed counters available
        :param general_counters: Number of general counters available
        """
        self.tdp = tdp
        self.freq_bclk = freq_bclk
        self.ratio_min = ratio_min
        self.ratio_base = ratio_base
        self.ratio_max = ratio_max
        self.fixed_counters = fixed_counters
        self.general_counters = general_counters

    def get_min_frequency(self) -> int:
        """
        Compute and return the CPU max efficiency frequency.
        :return: The CPU max efficiency frequency in MHz
        """
        return self.freq_bclk * self.ratio_min

    def get_base_frequency(self) -> int:
        """
        Compute and return the CPU base frequency.
        :return: The CPU base frequency in MHz
        """
        return self.freq_bclk * self.ratio_base

    def get_max_frequency(self) -> int:
        """
        Compute and return the CPU maximum frequency. (Turbo-Boost included)
        :return: The CPU maximum frequency in MHz
        """
        return self.freq_bclk * self.ratio_max

    def get_supported_frequencies(self) -> List[int]:
        """
        Compute the supported frequencies for this CPU.
        :return: A list of supported frequencies in MHz
        """
        return [freq for freq in range(self.get_min_frequency(), self.get_max_frequency() + 1, self.freq_bclk)]
