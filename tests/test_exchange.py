from __future__ import annotations

from eth_account import Account
from pydantic import SecretStr

from src.core.config import HyperliquidSettings
from src.exchange.hyperliquid_rest import AGENT_TYPES, PHANTOM_DOMAIN, HyperliquidRestClient


async def test_eip712_cache_initialized_on_construct() -> None:
    settings = HyperliquidSettings.from_env()
    client = HyperliquidRestClient(settings)

    assert client._cached_domain == PHANTOM_DOMAIN
    assert client._cached_types == AGENT_TYPES
    assert client._cached_domain["chainId"] == 1337
    assert client._cached_domain["verifyingContract"] == "0x0000000000000000000000000000000000000000"
    assert "Agent" in client._cached_types
    assert client._phantom_source == "a"


def test_sign_action_produces_signature() -> None:
    wallet = Account.create()
    settings = HyperliquidSettings.from_env().model_copy(
        update={"agent_private_key": SecretStr(wallet.key.hex())}
    )
    client = HyperliquidRestClient(settings)

    signed = client.sign_action({"type": "noop"}, 1_700_000_000_000)

    assert signed.nonce == 1_700_000_000_000
    assert all(key in signed.signature for key in ("r", "s", "v"))
