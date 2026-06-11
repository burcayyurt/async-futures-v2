"""Shared strategy signal contract.

The ``TradeSignal`` / ``SignalSide`` types are the interface between any strategy
engine (momentum, DVSLA, ...) and the execution layer (``order_router``). They
live here — independent of any single strategy module — so engines can be added
or retired without breaking the execution contract.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field


class SignalSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class TradeSignal(BaseModel):
    symbol: str
    side: SignalSide
    entry_mark_price: Decimal
    confidence: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    timestamp: datetime
    reason: str = ""

    model_config = {"arbitrary_types_allowed": True}


StrategySignal = TradeSignal


__all__ = ["SignalSide", "TradeSignal", "StrategySignal"]
