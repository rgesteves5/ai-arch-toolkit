"""Tests for _pricing.py — PricingRegistry."""

from __future__ import annotations

from pathlib import Path

from ai_arch_toolkit.core._pricing import ModelPricing, PricingRegistry, estimate_cost, pricing


class TestPricingRegistryDefaults:
    """The module-level singleton loads from _default_pricing.yaml."""

    def test_has_claude_models(self):
        assert pricing.has("claude-sonnet-4-20250514")
        assert pricing.has("claude-opus-4-20250514")

    def test_has_gpt_models(self):
        assert pricing.has("gpt-4o-2024-08-06")
        assert pricing.has("gpt-4o-mini-2024-07-18")

    def test_unknown_model(self):
        assert not pricing.has("unknown-model-v1")
        assert pricing.get("unknown-model-v1") is None

    def test_list_models_returns_sorted(self):
        models = pricing.list_models()
        assert models == sorted(models)
        assert len(models) > 10


class TestPricingRegistryGet:
    def test_exact_prefix_match(self):
        p = pricing.get("claude-sonnet-4-20250514")
        assert p is not None
        assert p.input == 3.0

    def test_longest_prefix_wins(self):
        # "claude-3-5-sonnet" is longer than "claude-3-sonnet"
        p = pricing.get("claude-3-5-sonnet-20241022")
        assert p is not None
        assert p.input == 3.0

    def test_gpt_4o_mini_before_gpt_4o(self):
        p = pricing.get("gpt-4o-mini-2024-07-18")
        assert p is not None
        assert p.input == 0.15


class TestPricingRegistryRegister:
    def test_register_custom_model(self):
        reg = PricingRegistry()
        reg.register("my-model", ModelPricing(input=1.0, output=2.0))
        p = reg.get("my-model-v1")
        assert p is not None
        assert p.input == 1.0

    def test_override_existing(self):
        reg = PricingRegistry()
        reg.register("claude-sonnet-4", ModelPricing(input=99.0, output=99.0))
        p = reg.get("claude-sonnet-4-20250514")
        assert p is not None
        assert p.input == 99.0

    def test_unregister(self):
        reg = PricingRegistry()
        reg.register("temp-model", ModelPricing(input=1.0, output=1.0))
        assert reg.has("temp-model-v1")
        reg.unregister("temp-model")
        assert not reg.has("temp-model-v1")


class TestPricingRegistryReset:
    def test_reset_clears_custom(self):
        reg = PricingRegistry()
        reg.register("custom", ModelPricing(input=1.0, output=1.0))
        reg.reset()
        assert not reg.has("custom-v1")
        # Defaults still there
        assert reg.has("claude-sonnet-4-20250514")


class TestEstimateCost:
    def test_known_model(self):
        cost, known = pricing.estimate_cost(
            "claude-sonnet-4-20250514", input_tokens=1000, output_tokens=500
        )
        expected = 3.0 * 1000 / 1_000_000 + 15.0 * 500 / 1_000_000
        assert abs(cost - expected) < 1e-10
        assert known is True

    def test_unknown_model(self):
        cost, known = pricing.estimate_cost("unknown-model", input_tokens=1000)
        assert cost == 0.0
        assert known is False

    def test_cache_tokens(self):
        cost, known = pricing.estimate_cost(
            "claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=0,
            cache_write_tokens=500,
            cache_read_tokens=200,
        )
        expected = 3.0 * 1000 / 1_000_000 + 3.75 * 500 / 1_000_000 + 0.30 * 200 / 1_000_000
        assert abs(cost - expected) < 1e-10
        assert known is True

    def test_cache_tokens_ignored_for_models_without_cache(self):
        # gpt-4o has no cache pricing (None) — cache tokens should not contribute
        cost, _ = pricing.estimate_cost(
            "gpt-4o-2024-08-06",
            input_tokens=1000,
            output_tokens=0,
            cache_write_tokens=500,
        )
        expected = 2.50 * 1000 / 1_000_000
        assert abs(cost - expected) < 1e-10

    def test_batch_pricing(self):
        cost, known = pricing.estimate_cost(
            "claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
            is_batch=True,
        )
        expected = 1.50 * 1000 / 1_000_000 + 7.50 * 500 / 1_000_000
        assert abs(cost - expected) < 1e-10
        assert known is True

    def test_batch_fallback_to_normal_pricing(self):
        # gpt-4o has no batch pricing (None) — should fall back to normal rates
        cost, known = pricing.estimate_cost(
            "gpt-4o-2024-08-06",
            input_tokens=1000,
            output_tokens=500,
            is_batch=True,
        )
        expected = 2.50 * 1000 / 1_000_000 + 10.0 * 500 / 1_000_000
        assert abs(cost - expected) < 1e-10
        assert cost > 0.0
        assert known is True


class TestConvenienceEstimateCost:
    def test_returns_just_float(self):
        cost = estimate_cost("claude-sonnet-4-20250514", input_tokens=1000, output_tokens=500)
        assert isinstance(cost, float)
        assert cost > 0

    def test_unknown_returns_zero(self):
        assert estimate_cost("unknown-model", input_tokens=1000) == 0.0


class TestLoad:
    def test_load_toml(self, tmp_path: Path):
        toml_content = "[my-custom-model]\ninput = 5.0\noutput = 10.0\n"
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(toml_content)

        reg = PricingRegistry()
        reg.load(toml_file)
        p = reg.get("my-custom-model-v1")
        assert p is not None
        assert p.input == 5.0
        assert p.output == 10.0
        assert p.cache_write is None

    def test_load_merges(self, tmp_path: Path):
        toml_content = "[custom]\ninput = 1.0\noutput = 2.0\n"
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(toml_content)

        reg = PricingRegistry()
        reg.load(toml_file)
        # Custom model loaded
        assert reg.has("custom-v1")
        # Defaults still present
        assert reg.has("claude-sonnet-4-20250514")


class TestModelPricingNone:
    def test_none_semantics(self):
        # Models without cache/batch have None, not 0.0
        p = pricing.get("gpt-4o-2024-08-06")
        assert p is not None
        assert p.cache_write is None
        assert p.cache_read is None
        assert p.batch_input is None
        assert p.batch_output is None

    def test_claude_has_cache_pricing(self):
        p = pricing.get("claude-sonnet-4-20250514")
        assert p is not None
        assert p.cache_write is not None
        assert p.cache_read is not None
