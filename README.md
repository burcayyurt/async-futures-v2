# async-futures-v2

Hyperliquid perpetual futures bot with momentum + open interest strategy, dry-run support, trade journal, and Telegram ops.

## Features

- WebSocket market data fan-out (strategy, positions, risk)
- Momentum + OI breakout strategy with BTC regime filter
- Position manager: trailing stop, break-even, kill switch
- Persistent `data/trades.jsonl` (closed trades) and `data/open_positions.json` (crash recovery)
- Telegram: entry/exit/heartbeat, `/stats`, `/positions`, `/help`

## Quick start

```bash
cp .env.example .env
# Edit .env (never commit .env)

pip install -r requirements.txt
python -m pytest -q
python -m src.main
```

## Docker

```bash
docker compose up -d --build
```

`./data` is mounted for journal and open position persistence.

## Security

- Keep `WALLET_ADDRESS`, `AGENT_PRIVATE_KEY`, and `TELEGRAM_BOT_TOKEN` in `.env` only
- Default mode is `BOT_DRY_RUN=true`
