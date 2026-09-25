# Copyright 2026 UniFi Insights contributors
"""Tests for the UniFi Site Manager API client."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.unifi_insights.api.auth import ApiKeyAuth
from custom_components.unifi_insights.api.base import _retry_after_seconds
from custom_components.unifi_insights.api.const import DEFAULT_RATE_LIMIT_RETRY_AFTER
from custom_components.unifi_insights.api.exceptions import UniFiResponseError
from custom_components.unifi_insights.api.site_manager import UniFiSiteManagerClient


class _Response:
    """Minimal aiohttp response replacement for transport tests."""

    status = 200

    def __init__(self, body: dict[str, Any]) -> None:
        """Store a JSON response body."""
        self._body = body

    async def __aenter__(self) -> Self:
        """Enter the async response context."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit the async response context."""

    async def text(self) -> str:
        """Return the serializable response body."""
        return "{}"

    async def json(self) -> dict[str, Any]:
        """Return the response body."""
        return self._body


class _Session:
    """Queued response transport recording requests made by the client."""

    closed = False

    def __init__(self, bodies: list[dict[str, Any]]) -> None:
        """Initialize queued JSON bodies."""
        self._bodies = iter(bodies)
        self.requests: list[dict[str, Any]] = []

    def request(self, method: str, url: object, **kwargs: Any) -> _Response:
        """Record a request and return the next configured response."""
        self.requests.append({"method": method, "url": str(url), **kwargs})
        return _Response(next(self._bodies))


def _client(session: _Session) -> UniFiSiteManagerClient:
    """Create a Site Manager client using a recording transport."""
    return UniFiSiteManagerClient(ApiKeyAuth("test-key"), session=session)  # type: ignore[arg-type]


async def test_list_devices_paginates_merges_host_groups_and_encodes_host_ids() -> None:
    """Device pages preserve repeated host filters and merge the same host."""
    session = _Session(
        [
            {
                "data": [
                    {"hostId": "host-a", "hostName": "A", "devices": [{"id": "1"}]}
                ],
                "nextToken": "page-2",
            },
            {
                "data": [{"hostId": "host-a", "devices": [{"id": "2"}]}],
            },
        ]
    )

    devices = await _client(session).list_devices(["host-a", "host-b"])

    assert devices == [
        {
            "hostId": "host-a",
            "hostName": "A",
            "devices": [{"id": "1"}, {"id": "2"}],
        }
    ]
    assert session.requests[0]["url"] == "https://api.ui.com/v1/devices"
    assert session.requests[0]["params"] == {"hostIds[]": ["host-a", "host-b"]}
    assert session.requests[1]["params"] == {
        "hostIds[]": ["host-a", "host-b"],
        "nextToken": "page-2",
    }
    assert session.requests[0]["headers"]["X-API-Key"] == "test-key"


async def test_list_hosts_rejects_a_repeated_pagination_token() -> None:
    """A repeated token fails instead of making an unbounded request loop."""
    session = _Session(
        [
            {"data": [{"id": "host-a"}], "nextToken": "again"},
            {"data": [{"id": "host-b"}], "nextToken": "again"},
        ]
    )

    with pytest.raises(UniFiResponseError) as error:
        await _client(session).list_hosts()
    assert "repeated nextToken" in error.value.args[0]


async def test_null_devices_array_is_an_empty_host_group() -> None:
    """The documented nullable devices field does not discard inventory."""
    session = _Session([{"data": [{"hostId": "host-a", "devices": None}]}])

    assert await _client(session).list_devices() == [
        {"hostId": "host-a", "devices": []}
    ]


async def test_malformed_devices_array_is_rejected() -> None:
    """An empty object does not silently masquerade as an empty device list."""
    session = _Session([{"data": [{"hostId": "host-a", "devices": {}}]}])

    with pytest.raises(UniFiResponseError) as error:
        await _client(session).list_devices()
    assert "malformed devices" in error.value.args[0]


async def test_metrics_and_sd_wan_validate_envelopes_and_format_timestamps() -> None:
    """Non-paginated data endpoints validate their envelopes and query format."""
    session = _Session(
        [
            {"data": [{"hostId": "host-a", "periods": []}]},
            {"data": [{"id": "config-a", "type": "sdwan-hbsp"}]},
            {"data": {"id": "invalid"}},
        ]
    )
    client = _client(session)

    metrics = await client.get_isp_metrics(
        datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        datetime(2026, 1, 2, 4, 5, 6, tzinfo=UTC),
    )
    configs = await client.list_sd_wan_configs()

    assert metrics == [{"hostId": "host-a", "periods": []}]
    assert configs == [{"id": "config-a", "type": "sdwan-hbsp"}]
    assert session.requests[0]["url"] == "https://api.ui.com/v1/isp-metrics/5m"
    assert session.requests[0]["params"] == {
        "beginTimestamp": "2026-01-02T03:04:05Z",
        "endTimestamp": "2026-01-02T04:05:06Z",
    }
    with pytest.raises(UniFiResponseError) as error:
        await client.list_sd_wan_configs()
    assert "malformed data" in error.value.args[0]


def test_retry_after_parser_handles_fractional_and_invalid_headers() -> None:
    """Cloud rate limiting keeps its delay even for fractional headers."""
    assert _retry_after_seconds("5.372786998") == 6
    assert _retry_after_seconds("invalid") == DEFAULT_RATE_LIMIT_RETRY_AFTER


async def test_site_manager_response_bodies_are_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cloud inventory stays private in debug and non-JSON warning logs."""
    client = _client(_Session([]))
    response = MagicMock(status=200, method="GET")
    response.url.path = "/v1/hosts"
    response.text = AsyncMock(return_value='{"id":"private-host-id"}')
    response.json = AsyncMock(return_value={"data": []})

    with caplog.at_level(
        logging.DEBUG, logger="custom_components.unifi_insights.api.base"
    ):
        await client._handle_response(response)
        response.json.side_effect = ValueError("invalid JSON")
        with pytest.raises(UniFiResponseError):
            await client._handle_response(response)

    assert "private-host-id" not in caplog.text
    assert "[Site Manager response omitted]" in caplog.text
