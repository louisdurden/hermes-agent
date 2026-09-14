from hermes_cli.goa_config import (
    DEFAULT_GOA_META,
    DEFAULT_GOA_PROFILES,
    decode_goa_turn,
    encode_goa_turn,
    get_goa_config,
    is_goa_enabled,
)


def test_goa_defaults_fail_closed_and_bounded():
    config = get_goa_config({})
    assert config == {
        "enabled": False,
        "top_k": 3,
        "rounds": 1,
        "edge_threshold": 0.0,
        "profiles": DEFAULT_GOA_PROFILES,
        "meta": DEFAULT_GOA_META,
        "token_budget": 7000,
        "invocation_budget": 12,
    }
    assert is_goa_enabled({}) is False


def test_goa_requires_literal_true():
    assert is_goa_enabled({"goa": {"enabled": True}}) is True
    assert is_goa_enabled({"goa": {"enabled": "true"}}) is False
    assert is_goa_enabled({"goa": {"enabled": 1}}) is False


def test_goa_normalization_enforces_bounded_contract():
    config = get_goa_config(
        {
            "goa": {
                "enabled": True,
                "top_k": 99,
                "rounds": 99,
                "token_budget": 999999,
                "invocation_budget": 999,
                "profiles": ["fleetseo", "unknown"],
                "meta": {"model": "custom", "untrusted": "discard"},
            }
        }
    )
    assert config["enabled"] is True
    assert config["top_k"] == 3
    assert config["rounds"] == 1
    assert config["token_budget"] == 7000
    assert config["invocation_budget"] == 12
    assert config["profiles"] == ("fleetseo",)
    assert config["meta"]["model"] == "custom"
    assert "untrusted" not in config["meta"]


def test_encoded_turn_round_trips_normalized_config():
    encoded = encode_goa_turn("question", {"enabled": True, "profiles": ["fleetseo"]})
    prompt, config = decode_goa_turn(encoded)
    assert prompt == "question"
    assert config is not None
    assert config["enabled"] is True
    assert config["profiles"] == ("fleetseo",)


def test_malformed_marker_is_plain_user_input():
    message = "__HERMES_GOA_TURN_V1__not-base64"
    assert decode_goa_turn(message) == (message, None)
