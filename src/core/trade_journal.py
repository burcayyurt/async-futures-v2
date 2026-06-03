from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.config import HyperliquidSettings
from src.strategy.momentum_oi import SignalSide

if TYPE_CHECKING:
    from src.execution.position_manager import ManagedPosition

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClosedTradeRecord:
    symbol: str
    side: str
    entry_px: Decimal
    exit_px: Decimal
    size: Decimal
    pnl_usd: Decimal
    roe_pct: Decimal
    exit_reason: str
    opened_at: datetime
    closed_at: datetime
    dry_run: bool

    def to_json_line(self) -> str:
        payload = {
            "symbol": self.symbol,
            "side": self.side,
            "entry_px": str(self.entry_px),
            "exit_px": str(self.exit_px),
            "size": str(self.size),
            "pnl_usd": str(self.pnl_usd.quantize(Decimal("0.01"))),
            "roe_pct": str(self.roe_pct.quantize(Decimal("0.1"))),
            "exit_reason": self.exit_reason,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat(),
            "dry_run": self.dry_run,
        }
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClosedTradeRecord:
        return cls(
            symbol=str(data["symbol"]).strip().upper(),
            side=str(data["side"]).lower(),
            entry_px=Decimal(str(data["entry_px"])),
            exit_px=Decimal(str(data["exit_px"])),
            size=Decimal(str(data["size"])),
            pnl_usd=Decimal(str(data["pnl_usd"])),
            roe_pct=Decimal(str(data["roe_pct"])),
            exit_reason=str(data["exit_reason"]),
            opened_at=datetime.fromisoformat(str(data["opened_at"])),
            closed_at=datetime.fromisoformat(str(data["closed_at"])),
            dry_run=bool(data.get("dry_run", False)),
        )


@dataclass(frozen=True, slots=True)
class PeriodStats:
    trade_count: int
    wins: int
    losses: int
    win_rate_pct: Decimal
    net_pnl_usd: Decimal
    lookback_days: int


def calc_roe_pct(
    *,
    side: SignalSide,
    entry_px: Decimal,
    exit_px: Decimal,
    leverage: int,
) -> Decimal:
    if entry_px <= 0:
        return Decimal("0")
    if side == SignalSide.LONG:
        price_move_pct = ((exit_px - entry_px) / entry_px) * Decimal("100")
    else:
        price_move_pct = ((entry_px - exit_px) / entry_px) * Decimal("100")
    return price_move_pct * Decimal(leverage)


class TradeJournal:
    """Append-only closed-trade journal backed by JSONL."""

    def __init__(self, settings: HyperliquidSettings, *, path: Path | None = None) -> None:
        self._settings = settings
        self._path = path or Path(settings.trade_journal_path)

    @property
    def path(self) -> Path:
        return self._path

    def record_closed_trade(
        self,
        position: ManagedPosition,
        exit_px: Decimal,
        pnl_usd: Decimal,
        roe_pct: Decimal,
        exit_reason: str,
        *,
        closed_at: datetime | None = None,
    ) -> None:
        closed_ts = closed_at or datetime.now(timezone.utc)
        if closed_ts.tzinfo is None:
            closed_ts = closed_ts.replace(tzinfo=timezone.utc)

        opened_at = position.opened_at
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)

        record = ClosedTradeRecord(
            symbol=position.coin.strip().upper(),
            side=position.side.value,
            entry_px=position.entry_px,
            exit_px=exit_px,
            size=position.size,
            pnl_usd=pnl_usd,
            roe_pct=roe_pct,
            exit_reason=exit_reason,
            opened_at=opened_at,
            closed_at=closed_ts,
            dry_run=self._settings.bot_dry_run,
        )
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(record.to_json_line() + "\n")
        except OSError:
            logger.warning("Failed to append closed trade to %s", self._path, exc_info=True)

    def load_closed_trades(self, *, since: datetime) -> list[ClosedTradeRecord]:
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        if not self._path.exists():
            return []

        trades: list[ClosedTradeRecord] = []
        try:
            with self._path.open(encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        record = ClosedTradeRecord.from_dict(json.loads(stripped))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        logger.warning("Skipping malformed trade journal line %s in %s", line_no, self._path)
                        continue
                    closed_at = record.closed_at
                    if closed_at.tzinfo is None:
                        closed_at = closed_at.replace(tzinfo=timezone.utc)
                    if closed_at >= since:
                        trades.append(record)
        except OSError:
            logger.warning("Failed to read trade journal from %s", self._path, exc_info=True)
        return trades

    def load_all_closed_trades(self) -> list[ClosedTradeRecord]:
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        return self.load_closed_trades(since=epoch)

    def period_stats(self, *, days: int = 7, now: datetime | None = None) -> PeriodStats:
        ts = now or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        since = ts - timedelta(days=days)
        trades = self.load_closed_trades(since=since)

        wins = sum(1 for trade in trades if trade.pnl_usd > 0)
        losses = sum(1 for trade in trades if trade.pnl_usd < 0)
        trade_count = len(trades)
        net_pnl = sum((trade.pnl_usd for trade in trades), Decimal("0"))

        if trade_count == 0:
            win_rate = Decimal("0")
        else:
            win_rate = (Decimal(wins) / Decimal(trade_count)) * Decimal("100")

        return PeriodStats(
            trade_count=trade_count,
            wins=wins,
            losses=losses,
            win_rate_pct=win_rate.quantize(Decimal("0.1")),
            net_pnl_usd=net_pnl.quantize(Decimal("0.01")),
            lookback_days=days,
        )
