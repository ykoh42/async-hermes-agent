from decimal import Decimal

from agent.models_dev import ModelInfo
from hermes_cli.model_cost_guard import expensive_model_warning


def test_no_warning_when_known_prices_are_at_threshold():
    info = ModelInfo(
        id="edge/model",
        name="edge/model",
        family="",
        provider_id="test",
        cost_input=20.0,
        cost_output=100.0,
    )

    assert expensive_model_warning("edge/model", provider="test", model_info=info) is None






def test_openai_gpt55_pro_warns_for_known_pricing():
    info = ModelInfo(
        id="openai/gpt-5.5-pro",
        name="openai/gpt-5.5-pro",
        family="gpt-5",
        provider_id="nous",
        cost_input=25.0,
        cost_output=125.0,
    )

    warning = expensive_model_warning(
        "openai/gpt-5.5-pro", provider="nous", model_info=info
    )

    assert warning is not None
    assert warning.input_cost_per_million == Decimal("25.000000")
    assert warning.output_cost_per_million == Decimal("125.000000")
    assert "did you mean to select openai/gpt-5.5?" in warning.message
