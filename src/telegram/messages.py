from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import TYPE_CHECKING

from src.core.trade_journal import PeriodStats
from src.strategy.momentum_oi import SignalSide

if TYPE_CHECKING:
    from src.execution.position_manager import ManagedPosition


def _format_price(value: Decimal) -> str:
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"${text}"


def _format_pnl(value: Decimal) -> str:
    sign = "+" if value >= 0 else "-"
    amount = abs(value).quantize(Decimal("0.01"))
    text = format(amount.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{sign}${text}"


def _format_open_duration(opened_at: datetime, now: datetime) -> str:
    if opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    delta = now - opened_at
    if delta.total_seconds() < 0:
        delta = timedelta(0)
    total_seconds = int(delta.total_seconds())
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def format_stats_message(stats: PeriodStats, *, all_time_count: int) -> str:
    if stats.trade_count == 0:
        return (
            f"📊 *İstatistikler ({stats.lookback_days} Gün)*\n"
            f"Henüz kapalı işlem yok.\n"
            f"Toplam kayıtlı kapalı işlem: {all_time_count}"
        )
    return (
        f"📊 *İstatistikler ({stats.lookback_days} Gün)*\n"
        f"İşlem: {stats.trade_count} | Kazanan: {stats.wins} | Kaybeden: {stats.losses}\n"
        f"Win Rate: %{stats.win_rate_pct} | Net PnL: {_format_pnl(stats.net_pnl_usd)}\n"
        f"Toplam kayıtlı kapalı işlem: {all_time_count}"
    )


def format_positions_message(positions: list[ManagedPosition], *, now: datetime | None = None) -> str:
    ts = now or datetime.now(timezone.utc)
    if not positions:
        return "📂 *Açık Pozisyonlar*\nŞu an açık pozisyon yok."

    lines = [f"📂 *Açık Pozisyonlar ({len(positions)})*"]
    for position in sorted(positions, key=lambda item: item.coin):
        side_label = "LONG" if position.side == SignalSide.LONG else "SHORT"
        duration = _format_open_duration(position.opened_at, ts)
        lines.append(
            f"{position.coin} {side_label} | Giriş {_format_price(position.entry_px)} | "
            f"Stop {_format_price(position.stop_px)} | {duration}"
        )
    return "\n".join(lines)


def format_help_message() -> str:
    return (
        "🤖 *Telegram Komutları*\n"
        "/stats — Win rate ve PnL özeti (7 gün)\n"
        "/positions — Açık pozisyon listesi\n"
        "/help — Bu mesaj"
    )
