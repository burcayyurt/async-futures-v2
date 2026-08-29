#!/usr/bin/env bash
# Read the remote bot's logs without remembering docker incantations.
#
# The container's logs are capped (10 MB x 5 files) by docker-compose.yml, so
# this reaches back a few days at most. Anything older has to come from the
# trade journal, which is the durable record — logs are for "what is it doing
# right now", the journal is for "what did it do".
#
# Usage:  BOT_HOST=root@1.2.3.4 scripts/logs.sh [tail|follow|faults|status]
set -euo pipefail

HOST="${BOT_HOST:?set BOT_HOST, e.g. BOT_HOST=root@66.42.36.198}"
MODE="${1:-status}"

# Lines that mean something is actually wrong. "Stale mark gap" and the missing
# WALLET_ADDRESS warning are expected in dry-run and would drown the signal.
FAULTS='rejected by exchange|Reconciliation|orphan|[Kk]ill.?switch|[Dd]rawdown|CRITICAL|Reconnecting in [0-9]{2,}|panic|Position adopted|Failed routing'
NOISE='Stale mark gap|WALLET_ADDRESS not set'

case "$MODE" in
  tail)
    ssh "$HOST" "docker logs --tail 120 futures_v2_bot 2>&1"
    ;;
  follow)
    ssh "$HOST" "docker logs -f --since 1m futures_v2_bot 2>&1"
    ;;
  faults)
    # Empty output is the good outcome.
    ssh "$HOST" "docker logs --since 24h futures_v2_bot 2>&1 \
      | grep -E '$FAULTS' | grep -vE '$NOISE' || echo 'son 24 saatte olay yok'"
    ;;
  status)
    ssh "$HOST" bash -s <<'REMOTE'
set -uo pipefail
cd /opt/async-futures-v2
echo "=== konteyner ==="
docker ps --filter name=futures_v2_bot --format '{{.Status}}'
docker stats --no-stream --format 'CPU {{.CPUPerc}}  RAM {{.MemUsage}}' futures_v2_bot
echo "=== yapilandirma ==="
docker logs futures_v2_bot 2>&1 | grep -m1 'Active config_id' || echo '(config_id satiri log penceresinden dusmus)'
echo "=== islemler ==="
if [ -f data/trades.jsonl ]; then
  wc -l < data/trades.jsonl | xargs echo 'toplam kapanan islem:'
  tail -1 data/trades.jsonl | python3 -c 'import json,sys; r=json.load(sys.stdin); print(f"son: {r[\"symbol\"]} {r[\"side\"]} {r[\"net_pnl_usd\"]}$ {r[\"exit_reason\"]} @ {r[\"closed_at\"][:19]}")'
else
  echo '(henuz kapanan islem yok)'
fi
echo "=== kayit ve disk ==="
du -sh data/recordings 2>/dev/null || echo '(kayit yok)'
ls -1 data/recordings 2>/dev/null | tail -3
df -h / | awk 'NR==2{print "disk: "$3" kullanilan / "$2" ("$5")"}'
echo "=== son aktivite ==="
# The summary lands every five minutes, so a freshly started bot has none yet;
# that is not a failure and must not colour this command's exit status.
docker logs --tail 400 futures_v2_bot 2>&1 | grep -E 'Activity summary' | tail -1 | cut -c1-240   || echo '(ozet satiri henuz yok; bot 5 dakikadan yeni)'
REMOTE
    ;;
  *)
    echo "bilinmeyen mod: $MODE (tail|follow|faults|status)" >&2
    exit 1
    ;;
esac
