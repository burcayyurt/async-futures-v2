# Running the bot on a remote host

## Why the bot is moving at all

Not for latency. That was measured and dismissed: the workstation is ~258 ms
from the Hyperliquid origin and a Tokyo host is 14 ms, but the price gap between
a signal and the market is
already ~14 bps at the first tick after the signal and does not grow over the
next half second (`scripts/latency_cost.py`, n=362). A host in Tokyo would
arrive to find the same gap. The entry is behind the market because of how it is
quoted, not because of where it is quoted from.

The reason is uptime. Between 2026-07-31 and 2026-08-29 the bot lost 13.7 hours
to one gap and 25 hours to another, both of them a workstation being turned off,
and Docker Desktop stopped four more times in a single afternoon. The research
programme needs roughly 600 trades — about three uninterrupted weeks — to
distinguish a +1.6 bps effect from zero. Uninterrupted is the operative word,
and it is the one thing a personal machine cannot promise.

## Sizing

The container uses 0.22% CPU and 61 MB of RAM, so the smallest instance any
provider sells is enough compute. Disk is the real constraint, and only because
of recordings: ~600 MB a day raw, ~65 MB after `scripts/compact_recordings.py`
(measured 9-12x on real data). A 40 GB disk holds years of compacted history.

Do not copy the existing archive up. The 2.5 GB of history lives here; the host
starts empty and its output flows the other way.

On a 1 GB instance, add swap before the first build — `pip install` of the
dependency set can exhaust memory where the running bot never would:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## First-time setup on the host

```bash
sudo mkdir -p /opt/async-futures-v2 && sudo chown "$USER" /opt/async-futures-v2
git clone https://github.com/burcayyurt/async-futures-v2.git /opt/async-futures-v2
```

Then, from this machine, once — `.env` holds the Telegram token and later the
agent key, so it is never in git and never in CI:

```bash
scp .env "$BOT_HOST:/opt/async-futures-v2/.env"
```

Nightly compaction, or the disk fills at ~600 MB a day:

```cron
5 3 * * * cd /opt/async-futures-v2 && /usr/bin/python3 -m scripts.compact_recordings
```

## Day to day

```bash
export BOT_HOST=user@your.host

scripts/deploy.sh            # test, push-check, rebuild, print the live config_id
scripts/pull_data.sh         # bring journal + compacted recordings back here
```

`deploy.sh` refuses to run with a dirty tree or unpushed commits, because a host
that cannot be reproduced from a pushed ref is a host nobody can reason about.
It ends by printing the `config_id` the container came up on — check it against
what you intended, since every observation window is attributed by that id.

## What still lives here

Analysis. The sweeps read the full history and take tens of minutes of CPU;
that is what a workstation is for. The host records and trades, this machine
studies, and `pull_data.sh` is the seam between them.

## Not yet done

- Off-host backup of the recordings. They cannot be re-downloaded from anywhere,
  so a dead disk is a permanent loss. Object storage with free egress
  (Cloudflare R2, Backblaze B2) costs cents at 2.5 GB; S3 works but charges to
  read your own data back.
- Live trading from a remote host puts `AGENT_PRIVATE_KEY` on a machine you do
  not physically control. Worth a separate decision, not a side effect of this
  move.
