"""Tests for the Home Assistant tool module.

Tests real logic: entity filtering, payload building, response parsing,
handler validation, and availability gating.
"""

import inspect
import json
from unittest.mock import AsyncMock, patch

import pytest

from tools.homeassistant_tool import (
    HA_CALL_SERVICE_SCHEMA,
    HA_GET_STATE_SCHEMA,
    HA_LIST_ENTITIES_SCHEMA,
    HA_LIST_SERVICES_SCHEMA,
    _BLOCKED_DOMAINS,
    _ENTITY_ID_RE,
    _SERVICE_NAME_RE,
    _build_service_payload,
    _check_ha_available,
    _filter_and_summarize,
    _get_headers,
    _handle_call_service,
    _handle_get_state,
    _handle_list_entities,
    _handle_list_services,
    _parse_service_response,
)


# ---------------------------------------------------------------------------
# Sample HA state data (matches real HA /api/states response shape)
# ---------------------------------------------------------------------------

SAMPLE_STATES = [
    {"entity_id": "light.bedroom", "state": "on", "attributes": {"friendly_name": "Bedroom Light", "brightness": 200}},
    {"entity_id": "light.kitchen", "state": "off", "attributes": {"friendly_name": "Kitchen Light"}},
    {"entity_id": "switch.fan", "state": "on", "attributes": {"friendly_name": "Living Room Fan"}},
    {"entity_id": "sensor.temperature", "state": "22.5", "attributes": {"friendly_name": "Kitchen Temperature", "unit_of_measurement": "C"}},
    {"entity_id": "climate.thermostat", "state": "heat", "attributes": {"friendly_name": "Main Thermostat", "current_temperature": 21}},
    {"entity_id": "binary_sensor.motion", "state": "off", "attributes": {"friendly_name": "Hallway Motion"}},
    {"entity_id": "sensor.humidity", "state": "55", "attributes": {"friendly_name": "Bedroom Humidity", "area": "bedroom"}},
]


# ---------------------------------------------------------------------------
# Entity filtering and summarization
# ---------------------------------------------------------------------------


class TestFilterAndSummarize:
    def test_no_filters_returns_all(self):
        result = _filter_and_summarize(SAMPLE_STATES)
        assert result["count"] == 7
        ids = {e["entity_id"] for e in result["entities"]}
        assert "light.bedroom" in ids
        assert "climate.thermostat" in ids

    def test_domain_filter_lights(self):
        result = _filter_and_summarize(SAMPLE_STATES, domain="light")
        assert result["count"] == 2
        for entity in result["entities"]:
            assert entity["entity_id"].startswith("light.")

    def test_missing_attributes_handled(self):
        states = [{"entity_id": "light.x", "state": "on"}]
        result = _filter_and_summarize(states)
        assert result["count"] == 1
        assert result["entities"][0]["friendly_name"] == ""


# ---------------------------------------------------------------------------
# Service payload building
# ---------------------------------------------------------------------------


class TestBuildServicePayload:
    def test_entity_id_only(self):
        payload = _build_service_payload(entity_id="light.bedroom")
        assert payload == {"entity_id": "light.bedroom"}

    def test_entity_id_param_takes_precedence_over_data(self):
        payload = _build_service_payload(
            entity_id="light.a",
            data={"entity_id": "light.b"},
        )
        assert payload["entity_id"] == "light.a"


# ---------------------------------------------------------------------------
# Service response parsing
# ---------------------------------------------------------------------------


class TestParseServiceResponse:
    def test_list_response_extracts_entities(self):
        ha_response = [
            {"entity_id": "light.bedroom", "state": "on", "attributes": {}},
            {"entity_id": "light.kitchen", "state": "on", "attributes": {}},
        ]
        result = _parse_service_response("light", "turn_on", ha_response)
        assert result["success"] is True
        assert result["service"] == "light.turn_on"
        assert len(result["affected_entities"]) == 2
        assert result["affected_entities"][0]["entity_id"] == "light.bedroom"

    def test_service_name_format(self):
        result = _parse_service_response("climate", "set_temperature", [])
        assert result["service"] == "climate.set_temperature"


# ---------------------------------------------------------------------------
# Handler validation (no mocks - these paths don't reach the network)
# ---------------------------------------------------------------------------


class TestHandlerValidation:
    @pytest.mark.asyncio
    async def test_get_state_missing_entity_id(self):
        result = json.loads(await _handle_get_state({}))
        assert "error" in result
        assert "entity_id" in result["error"]

    @pytest.mark.asyncio
    async def test_call_service_empty_strings(self):
        result = json.loads(await _handle_call_service({"domain": "", "service": ""}))
        assert "error" in result


# ---------------------------------------------------------------------------
# Handler JSON return contracts
# ---------------------------------------------------------------------------


class TestHandlerReturns:
    @pytest.mark.asyncio
    @patch("tools.homeassistant_tool._async_list_entities", new_callable=AsyncMock)
    async def test_list_entities_wraps_result(self, mock_list_entities):
        mock_list_entities.return_value = {"count": 0, "entities": []}
        result = json.loads(await _handle_list_entities({"domain": "light"}))
        assert result == {"result": {"count": 0, "entities": []}}
        mock_list_entities.assert_awaited_once_with(domain="light", area=None)

    @pytest.mark.asyncio
    @patch("tools.homeassistant_tool._async_get_state", new_callable=AsyncMock)
    async def test_get_state_wraps_result(self, mock_get_state):
        mock_get_state.return_value = {"entity_id": "light.test", "state": "on"}
        result = json.loads(await _handle_get_state({"entity_id": "light.test"}))
        assert result == {"result": {"entity_id": "light.test", "state": "on"}}
        mock_get_state.assert_awaited_once_with("light.test")

    @pytest.mark.asyncio
    @patch("tools.homeassistant_tool._async_list_services", new_callable=AsyncMock)
    async def test_list_services_wraps_result(self, mock_list_services):
        mock_list_services.return_value = {"count": 0, "domains": []}
        result = json.loads(await _handle_list_services({"domain": "light"}))
        assert result == {"result": {"count": 0, "domains": []}}
        mock_list_services.assert_awaited_once_with(domain="light")


# ---------------------------------------------------------------------------
# Security: domain blocklist
# ---------------------------------------------------------------------------


class TestDomainBlocklist:
    """Verify dangerous HA service domains are blocked."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("domain", sorted(_BLOCKED_DOMAINS))
    async def test_blocked_domain_rejected(self, domain):
        result = json.loads(await _handle_call_service({
            "domain": domain, "service": "any_service"
        }))
        assert "error" in result
        assert "blocked" in result["error"].lower()

    @pytest.mark.asyncio
    @patch("tools.homeassistant_tool._async_call_service", new_callable=AsyncMock)
    async def test_safe_domain_not_blocked(self, mock_call_service):
        """Safe domains like ``light`` reach the service-call layer."""
        mock_call_service.return_value = {"success": True}
        result = json.loads(await _handle_call_service({
            "domain": "light", "service": "turn_on", "entity_id": "light.test"
        }))
        assert result["result"]["success"] is True
        mock_call_service.assert_awaited_once_with(
            "light",
            "turn_on",
            "light.test",
            None,
        )

    def test_blocked_domains_include_shell_command(self):
        assert "shell_command" in _BLOCKED_DOMAINS

    def test_blocked_domains_include_hassio(self):
        assert "hassio" in _BLOCKED_DOMAINS


# ---------------------------------------------------------------------------
# Security: entity_id validation
# ---------------------------------------------------------------------------


class TestEntityIdValidation:
    """Verify entity_id format validation prevents path traversal."""

    def test_valid_entity_id_accepted(self):
        assert _ENTITY_ID_RE.match("light.bedroom")
        assert _ENTITY_ID_RE.match("sensor.temperature_1")
        assert _ENTITY_ID_RE.match("binary_sensor.motion")
        assert _ENTITY_ID_RE.match("climate.main_thermostat")

    def test_path_traversal_rejected(self):
        assert _ENTITY_ID_RE.match("../../config") is None
        assert _ENTITY_ID_RE.match("light/../../../etc/passwd") is None
        assert _ENTITY_ID_RE.match("../api/config") is None

    @pytest.mark.asyncio
    @patch("tools.homeassistant_tool._async_call_service", new_callable=AsyncMock)
    async def test_call_service_allows_no_entity_id(self, mock_call_service):
        """Some services (like scene.turn_on) don't need entity_id."""
        mock_call_service.return_value = {"success": True}
        result = json.loads(await _handle_call_service({
            "domain": "scene", "service": "turn_on"
        }))
        assert result["result"]["success"] is True
        mock_call_service.assert_awaited_once_with(
            "scene",
            "turn_on",
            None,
            None,
        )


# ---------------------------------------------------------------------------
# String-data deserialization (XML tool calling workaround)
# ---------------------------------------------------------------------------


class TestCallServiceStringData:
    """data param may arrive as a JSON string (XML tool calling mode)."""

    @pytest.mark.asyncio
    @patch("tools.homeassistant_tool._async_call_service", new_callable=AsyncMock)
    async def test_string_data_deserialized(self, mock_call_service):
        """JSON string data is parsed into a dict before dispatch."""
        mock_call_service.return_value = {"success": True}
        await _handle_call_service({
            "domain": "climate",
            "service": "set_hvac_mode",
            "entity_id": "climate.living_room",
            "data": '{"hvac_mode": "heat"}',
        })
        mock_call_service.assert_awaited_once_with(
            "climate",
            "set_hvac_mode",
            "climate.living_room",
            {"hvac_mode": "heat"},
        )

    @pytest.mark.asyncio
    @patch("tools.homeassistant_tool._async_call_service", new_callable=AsyncMock)
    async def test_empty_string_data_becomes_none(self, mock_call_service):
        """Empty/whitespace string data is treated as None."""
        mock_call_service.return_value = {"success": True}
        await _handle_call_service({
            "domain": "light",
            "service": "turn_on",
            "entity_id": "light.bedroom",
            "data": "   ",
        })
        mock_call_service.assert_awaited_once_with(
            "light",
            "turn_on",
            "light.bedroom",
            None,
        )


# ---------------------------------------------------------------------------
# Security: domain/service name format validation
# ---------------------------------------------------------------------------


class TestServiceNameValidation:
    """Verify domain/service format validation prevents path traversal in URL."""

    def test_valid_domain_names(self):
        assert _SERVICE_NAME_RE.match("light")
        assert _SERVICE_NAME_RE.match("switch")
        assert _SERVICE_NAME_RE.match("climate")
        assert _SERVICE_NAME_RE.match("shell_command")
        assert _SERVICE_NAME_RE.match("media_player")

    def test_path_traversal_in_domain_rejected(self):
        assert _SERVICE_NAME_RE.match("../../api/config") is None
        assert _SERVICE_NAME_RE.match("light/../../../etc") is None
        assert _SERVICE_NAME_RE.match("../config") is None

    def test_path_traversal_in_service_rejected(self):
        assert _SERVICE_NAME_RE.match("../../api/config") is None
        assert _SERVICE_NAME_RE.match("turn_on/../../config") is None

    def test_blocked_domain_bypass_via_traversal_rejected(self):
        assert _SERVICE_NAME_RE.match("shell_command/../light") is None
        assert _SERVICE_NAME_RE.match("python_script/../scene") is None
        assert _SERVICE_NAME_RE.match("hassio/../automation") is None

    def test_special_chars_rejected(self):
        assert _SERVICE_NAME_RE.match("light;rm") is None
        assert _SERVICE_NAME_RE.match("light&cmd") is None
        assert _SERVICE_NAME_RE.match("light cmd") is None

    @pytest.mark.asyncio
    async def test_handler_rejects_traversal_domain(self):
        result = json.loads(await _handle_call_service({
            "domain": "../../api/config",
            "service": "turn_on",
        }))
        assert "error" in result
        assert "Invalid domain" in result["error"]

    @pytest.mark.asyncio
    async def test_handler_rejects_traversal_service(self):
        result = json.loads(await _handle_call_service({
            "domain": "light",
            "service": "../../api/config",
        }))
        assert "error" in result
        assert "Invalid service" in result["error"]


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


class TestCheckAvailable:
    def test_unavailable_without_token(self, monkeypatch):
        monkeypatch.delenv("HASS_TOKEN", raising=False)
        assert _check_ha_available() is False

    def test_empty_token_is_unavailable(self, monkeypatch):
        monkeypatch.setenv("HASS_TOKEN", "")
        assert _check_ha_available() is False

    def test_multiplex_scope_does_not_fall_back_to_another_profile(self, monkeypatch):
        from agent import secret_scope

        monkeypatch.setenv("HASS_TOKEN", "default-profile-token")
        secret_scope.set_multiplex_active(True)
        token = secret_scope.set_secret_scope({})
        try:
            assert _check_ha_available() is False
        finally:
            secret_scope.reset_secret_scope(token)
            secret_scope.set_multiplex_active(False)

    def test_multiplex_scope_supplies_profile_url_and_token(self, monkeypatch):
        from agent import secret_scope
        from tools.homeassistant_tool import _get_config

        monkeypatch.setattr("tools.homeassistant_tool._HASS_URL", "")
        monkeypatch.setattr("tools.homeassistant_tool._HASS_TOKEN", "")
        monkeypatch.setenv("HASS_URL", "http://default-profile:8123")
        monkeypatch.setenv("HASS_TOKEN", "default-profile-token")
        secret_scope.set_multiplex_active(True)
        token = secret_scope.set_secret_scope({
            "HASS_URL": "http://secondary-profile:8123/",
            "HASS_TOKEN": "secondary-profile-token",
        })
        try:
            assert _get_config() == (
                "http://secondary-profile:8123",
                "secondary-profile-token",
            )
        finally:
            secret_scope.reset_secret_scope(token)
            secret_scope.set_multiplex_active(False)


# ---------------------------------------------------------------------------
# Auth headers
# ---------------------------------------------------------------------------


class TestGetHeaders:
    def test_bearer_token_format(self, monkeypatch):
        monkeypatch.setattr("tools.homeassistant_tool._HASS_TOKEN", "my-secret-token")
        headers = _get_headers()
        assert headers["Authorization"] == "Bearer my-secret-token"
        assert headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# Schema and registry integration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_schema_names_arguments_and_required_fields(self):
        assert HA_LIST_ENTITIES_SCHEMA["name"] == "ha_list_entities"
        assert HA_LIST_ENTITIES_SCHEMA["parameters"]["required"] == []
        assert set(HA_LIST_ENTITIES_SCHEMA["parameters"]["properties"]) == {
            "domain", "area"
        }
        assert HA_GET_STATE_SCHEMA["name"] == "ha_get_state"
        assert HA_GET_STATE_SCHEMA["parameters"]["required"] == ["entity_id"]
        assert HA_LIST_SERVICES_SCHEMA["name"] == "ha_list_services"
        assert HA_LIST_SERVICES_SCHEMA["parameters"]["required"] == []
        assert HA_CALL_SERVICE_SCHEMA["name"] == "ha_call_service"
        assert HA_CALL_SERVICE_SCHEMA["parameters"]["required"] == [
            "domain", "service"
        ]
        assert set(HA_CALL_SERVICE_SCHEMA["parameters"]["properties"]) == {
            "domain", "service", "entity_id", "data"
        }

    def test_tools_registered_in_registry(self):
        from tools.registry import registry

        for name in (
            "ha_list_entities",
            "ha_get_state",
            "ha_list_services",
            "ha_call_service",
        ):
            entry = registry.get_entry(name)
            assert entry is not None
            assert entry.toolset == "homeassistant"
            assert inspect.iscoroutinefunction(entry.handler)
            assert entry.check_fn is _check_ha_available

    @pytest.mark.asyncio
    async def test_check_fn_includes_when_token_set(self, monkeypatch):
        from tools.registry import invalidate_check_fn_cache, registry

        monkeypatch.setenv("HASS_TOKEN", "test-token")
        invalidate_check_fn_cache()
        defs = await registry.get_definitions({
            "ha_list_entities",
            "ha_get_state",
            "ha_list_services",
            "ha_call_service",
        })
        assert len(defs) == 4
