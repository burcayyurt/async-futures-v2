from __future__ import annotations

import asyncio
import logging
from decimal import Decimal

import aiohttp

from src.core.config import HyperliquidSettings
from src.strategy.momentum_oi import SignalSide

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"

EXIT_REASON_LABELS = {
    "trailing_stop": "Trailing Stop",
    "stop_loss": "Stop Loss",
    "kill_switch": "Kill Switch",
    "timeout": "Timeout",
}


class TelegramNotifier:
    """Lightweight fire-and-forget Telegram notifier using aiohttp."""

    def __init__(self, settings: HyperliquidSettings) -> None:
        self._settings = settings
        self._token = settings.telegram_bot_token.get_secret_value().strip()
        self._chat_id = settings.telegram_chat_id.strip()
        self._session: aiohttp.ClientSession | None = None
        self._pending_tasks: set[asyncio.Task[None]] = set()

    @property
    def enabled(self) -> bool:
        return bool(self._token and self._chat_id)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _invalidate_session(self) -> None:
        """Close and discard the current session so the next call gets a fresh connection pool."""
        if self._session is not None and not self._session.closed:
            try:
                await self._session.close()
            except Exception:
                pass
        self._session = None

    def _dry_run_prefix(self) -> str:
        if self._settings.bot_dry_run:
            return "*🧪 [DRY RUN / SANAL İŞLEM]*\n"
        return ""

    @staticmethod
    def _format_price(value: Decimal) -> str:
        normalized = value.normalize()
        text = format(normalized, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return f"${text}"

    @staticmethod
    def _format_pnl(value: Decimal) -> str:
        sign = "+" if value >= 0 else "-"
        amount = abs(value).quantize(Decimal("0.01"))
        text = format(amount.normalize(), "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return f"{sign}${text}"

    @staticmethod
    def _format_roe(roe_pct: Decimal) -> str:
        sign = "+" if roe_pct >= 0 else "-"
        amount = abs(roe_pct).quantize(Decimal("0.1"))
        text = format(amount.normalize(), "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return f"({sign}%{text} ROE)"

    @staticmethod
    def _format_exit_reason(exit_reason: str) -> str:
        return EXIT_REASON_LABELS.get(exit_reason, exit_reason.replace("_", " ").title())

    @staticmethod
    def _format_win_rate(win_rate_pct: Decimal) -> str:
        text = format(win_rate_pct.quantize(Decimal("0.1")).normalize(), "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return f"%{text}"

    def notify(self, text: str) -> None:
        if not self.enabled:
            return
        task = asyncio.create_task(self._send(text))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def send_message(self, text: str) -> None:
        if not self.enabled:
            return
        await self._send(text)

    async def _send(self, text: str) -> None:
        url = f"{TELEGRAM_API_BASE}/bot{self._token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }
        try:
            session = await self._get_session()
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status >= 400:
                    body = await response.text()
                    logger.warning("Telegram sendMessage failed status=%s body=%s", response.status, body)
        except Exception:
            logger.exception("Telegram notification failed")
            await self._invalidate_session()

    def notify_entry(
        self,
        coin: str,
        side: SignalSide,
        entry_px: Decimal,
        size: Decimal,
    ) -> None:
        side_label = "LONG" if side == SignalSide.LONG else "SHORT"
        emoji = "🟢" if side == SignalSide.LONG else "🔴"
        qty_label = "Sanal Adet" if self._settings.bot_dry_run else "Adet"
        text = (
            f"{self._dry_run_prefix()}"
            f"{emoji} {side_label} Açıldı | Sembol: {coin.strip().upper()} | "
            f"Giriş: {self._format_price(entry_px)} | {qty_label}: {size.normalize()}"
        )
        self.notify(text)

    def notify_exit(
        self,
        coin: str,
        exit_px: Decimal,
        pnl_usd: Decimal,
        *,
        roe_pct: Decimal,
        exit_reason: str,
        side: SignalSide,
    ) -> None:
        del side
        text = (
            f"{self._dry_run_prefix()}"
            f"✅ İŞLEM KAPANDI | Sembol: {coin.strip().upper()} | "
            f"Çıkış: {self._format_price(exit_px)} | "
            f"Kâr/Zarar: {self._format_pnl(pnl_usd)} {self._format_roe(roe_pct)} | "
            f"Sebep: {self._format_exit_reason(exit_reason)}"
        )
        self.notify(text)

    def notify_kill_switch(self) -> None:
        self.notify("🚨 DİKKAT: KILL SWITCH TETİKLENDİ! Sanal kasa limiti aşıldı.")

    def notify_startup(
        self,
        *,
        symbol_count: int,
        recovered: list[str],
        journal_trade_count: int,
        engine: str | None = None,
    ) -> None:
        mode = "DRY RUN (Sanal)" if self._settings.bot_dry_run else "LIVE"
        engine_label = f" | Motor: {engine.upper()}" if engine else ""
        if recovered:
            recovered_line = f"Geri yüklenen pozisyon: {len(recovered)} ({', '.join(recovered)})"
        else:
            recovered_line = "Geri yüklenen pozisyon: 0"
        text = (
            f"✅ *HL Futures Bot Başladı*\n"
            f"Mod: {mode} | Coin: {symbol_count}{engine_label}\n"
            f"{recovered_line}\n"
            f"Kayıtlı kapalı işlem: {journal_trade_count}\n"
            f"Komutlar: /stats /positions /help"
        )
        self.notify(text)

    def notify_shutdown(self) -> None:
        prefix = "🧪 [DRY RUN]\n" if self._settings.bot_dry_run else ""
        self.notify(f"{prefix}🛑 Bot kapatılıyor...")

    def notify_heartbeat(
        self,
        open_symbols: list[str],
        *,
        win_rate_pct: Decimal,
        net_pnl_usd: Decimal,
        trade_count: int,
        lookback_days: int = 7,
    ) -> None:
        prefix = "🧪 [DRY RUN]\n" if self._settings.bot_dry_run else ""
        if open_symbols:
            positions_line = f"Açık Pozisyonlar: {len(open_symbols)} ({', '.join(open_symbols)})"
        else:
            positions_line = "Açık Pozisyonlar: 0"

        if trade_count == 0:
            stats_line = f"{lookback_days} Günlük: henüz kapalı işlem yok"
        else:
            stats_line = (
                f"{lookback_days} Günlük Win Rate: {self._format_win_rate(win_rate_pct)} | "
                f"Net PnL: {self._format_pnl(net_pnl_usd)}"
            )

        text = (
            f"{prefix}💓 [HEARTBEAT] Bot aktif olarak piyasayı dinliyor.\n"
            f"{positions_line} | {stats_line}"
        )
        self.notify(text)

    async def close(self) -> None:
        for task in self._pending_tasks:
            task.cancel()
        self._pending_tasks.clear()
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
