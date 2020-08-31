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

import logging
from typing import Dict

from powerapi.formula import FormulaActor
from powerapi.handler import PoisonPillMessageHandler
from powerapi.message import PoisonPillMessage
from powerapi.pusher import PusherActor
from powerapi.report import HWPCReport

from selfwatts.context import SelfWattsFormulaConfig, SelfWattsFormulaState
from selfwatts.handler import ReportHandler


class SelfWattsFormulaActor(FormulaActor):
    """
    This actor handle the reports for the SelfWatts formula.
    """

    def __init__(self, name: str, pushers: Dict[str, PusherActor], config: SelfWattsFormulaConfig):
        """
        Initialize new SelfWatts formula actor.
        :param name: Name of the actor
        :param pushers: Pusher actors
        :param config: Configuration of the formula
        """
        FormulaActor.__init__(self, name, pushers, logging.WARNING)
        self.state: SelfWattsFormulaState = SelfWattsFormulaState(self, pushers, config)

    def setup(self):
        """
        Setup the messages handlers.
        """
        FormulaActor.setup(self)
        self.add_handler(PoisonPillMessage, PoisonPillMessageHandler(self.state))
        self.add_handler(HWPCReport, ReportHandler(self.state))
