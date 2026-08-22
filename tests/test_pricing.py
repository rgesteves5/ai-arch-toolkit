"""Tests for _pricing.py — PricingRegistry."""

from __future__ import annotations

from pathlib import Path

from ai_arch_toolkit.core import ModelPricing, PricingRegistry, pricing
from ai_arch_toolkit.core._pricing import estimate_cost


class TestPricingRegistryDefaults:
    """The module-level singleton loads from _default_pricing.toml."""

    def test_has_claude_models(self):
        assert pricing.has("claude-opus-5")
        assert pricing.has("claude-sonnet-5")
        assert pricing.has("claude-fable-5")
        assert pricing.has("claude-opus-4-8")

    def test_has_claude_46_models(self):
        assert pricing.has("claude-opus-4-6-20260101")
        assert pricing.has("claude-sonnet-4-6-20260101")

    def test_has_claude_45_models(self):
        assert pricing.has("claude-opus-4-5-20260101")
        assert pricing.has("claude-haiku-4-5-20251001")

    def test_has_gpt_models(self):
        assert pricing.has("gpt-4o-2024-08-06")
        assert pricing.has("gpt-4o-mini-2024-07-18")

    def test_has_gpt41_models(self):
        assert pricing.has("gpt-4.1")
        assert pricing.has("gpt-4.1-mini")
        assert pricing.has("gpt-4.1-nano")

    def test_has_gpt5_models(self):
        assert pricing.has("gpt-5")
        assert pricing.has("gpt-5-mini")
        assert pricing.has("gpt-5-nano")
        assert pricing.has("gpt-5.2")
        assert pricing.has("gpt-5.2-pro")
        assert pricing.has("gpt-5-pro")
        assert pricing.has("gpt-5.6-sol")
        assert pricing.has("gpt-5.6-terra")
        assert pricing.has("gpt-5.6-luna")

    def test_has_gemini_models(self):
        assert pricing.has("gemini-3.6-flash")
        assert pricing.has("gemini-3.5-flash")
        assert pricing.has("gemini-3.5-flash-lite")
        assert pricing.has("gemini-3.1-pro-preview-20260301")
        assert pricing.has("gemini-3-flash-preview")
        assert pricing.has("gemini-2.5-flash")
        assert pricing.has("gemini-2.5-pro")
        assert pricing.has("gemini-2.0-flash-lite")

    def test_has_o_series_models(self):
        assert pricing.has("o3")
        assert pricing.has("o3-mini")
        assert pricing.has("o4-mini-deep-research")
        assert pricing.has("o1-pro")
        assert pricing.has("grok-3")

    def test_has_grok4_models(self):
        assert pricing.has("grok-4.5")
        assert pricing.has("grok-4.20-0309-reasoning")
        assert pricing.has("grok-build-0.1")
        assert pricing.has("grok-4")
        assert pricing.has("grok-4-1-fast-reasoning")
        assert pricing.has("grok-4-fast-reasoning")
        assert pricing.has("grok-code-fast-1")

    def test_grok_46_has_its_own_entry_not_the_grok_4_prefix(self):
        entry = pricing.get("grok-4.6")
        assert entry is not None
        assert entry.input == 2.0
        assert entry.output == 6.0
        assert entry.cache_read == 0.50
        assert entry != pricing.get("grok-4")

    def test_unknown_model(self):
        assert not pricing.has("unknown-model-v1")
        assert pricing.get("unknown-model-v1") is None

    def test_list_models_returns_sorted(self):
        models = pricing.list_models()
        assert models == sorted(models)
        assert len(models) > 10


class TestPricingRegistryGet:
    def test_exact_prefix_match(self):
        p = pricing.get("claude-sonnet-4-6-20260101")
        assert p is not None
        assert p.input == 3.0

    def test_longest_prefix_wins(self):
        # "grok-4.5" is longer than "grok-4"
        p = pricing.get("grok-4.5")
        assert p is not None
        assert p.input == 2.0  # grok-4.5, not grok-4's $1.25

    def test_gpt_4o_mini_before_gpt_4o(self):
        p = pricing.get("gpt-4o-mini-2024-07-18")
        assert p is not None
        assert p.input == 0.15

    def test_gpt56_variant_wins_over_gpt5(self):
        # "gpt-5.6-sol" is longer than "gpt-5"
        p = pricing.get("gpt-5.6-sol")
        assert p is not None
        assert p.input == 5.0  # gpt-5.6-sol, not gpt-5's $1.25

    def test_haiku_45_specific_prefix(self):
        p = pricing.get("claude-haiku-4-5-20251001")
        assert p is not None
        assert p.input == 1.0

    def test_o3_updated_pricing(self):
        p = pricing.get("o3")
        assert p is not None
        assert p.input == 2.0
        assert p.output == 8.0
        assert p.batch_input == 1.0
        assert p.cache_read == 0.50

    def test_gpt5_family_prefix_match(self):
        # gpt-5.2 is more specific than gpt-5
        p52 = pricing.get("gpt-5.2")
        p5 = pricing.get("gpt-5")
        assert p52 is not None
        assert p5 is not None
        assert p52.input == 1.75  # gpt-5.2
        assert p5.input == 1.25  # gpt-5 / gpt-5.1

    def test_gpt4o_has_cache_and_batch(self):
        p = pricing.get("gpt-4o-2024-08-06")
        assert p is not None
        assert p.cache_read == 1.25
        assert p.batch_input == 1.25
        assert p.batch_output == 5.0

    def test_gemini_25_pro_long_context(self):
        p = pricing.get("gemini-2.5-pro-latest")
        assert p is not None
        assert p.long_context_threshold == 200_000
        assert p.long_context_input == 2.50
        assert p.long_context_output == 15.0

    def test_gemini_20_flash_updated(self):
        p = pricing.get("gemini-2.0-flash")
        assert p is not None
        assert p.input == 0.15
        assert p.output == 0.60

    def test_grok_code_fast(self):
        p = pricing.get("grok-code-fast-1")
        assert p is not None
        assert p.input == 1.25
        assert p.output == 2.50
        assert p.cache_read == 0.20


class TestPricingRegistryRegister:
    def test_register_custom_model(self):
        reg = PricingRegistry()
        reg.register("my-model", ModelPricing(input=1.0, output=2.0))
        p = reg.get("my-model-v1")
        assert p is not None
        assert p.input == 1.0

    def test_override_existing(self):
        reg = PricingRegistry()
        reg.register("claude-sonnet-4-6", ModelPricing(input=99.0, output=99.0))
        p = reg.get("claude-sonnet-4-6-20260101")
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
        assert reg.has("claude-sonnet-4-6-20260101")


class TestEstimateCost:
    def test_known_model(self):
        cost = pricing.estimate_cost(
            "claude-sonnet-4-6-20260101", input_tokens=1000, output_tokens=500
        )
        expected = 3.0 * 1000 / 1_000_000 + 15.0 * 500 / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_unknown_model(self):
        cost = pricing.estimate_cost("unknown-model", input_tokens=1000)
        assert cost is None

    def test_cache_tokens(self):
        cost = pricing.estimate_cost(
            "claude-sonnet-4-6-20260101",
            input_tokens=1000,
            output_tokens=0,
            cache_write_tokens=500,
            cache_read_tokens=200,
        )
        expected = 3.0 * 1000 / 1_000_000 + 3.75 * 500 / 1_000_000 + 0.30 * 200 / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_cached_prompt_tokens_are_not_double_charged(self):
        # Provider prompt total: 1,000 = 800 uncached + 200 cache reads.
        cost = pricing.estimate_cost(
            "gpt-4o",
            input_tokens=800,
            output_tokens=0,
            cache_read_tokens=200,
        )
        expected = 2.50 * 800 / 1_000_000 + 1.25 * 200 / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_cache_tokens_ignored_for_models_without_cache(self):
        # gpt-4-turbo has no cache pricing (None) — cache tokens should not contribute
        cost = pricing.estimate_cost(
            "gpt-4-turbo-2024-04-09",
            input_tokens=1000,
            output_tokens=0,
            cache_write_tokens=500,
        )
        expected = 10.0 * 1000 / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_batch_pricing(self):
        cost = pricing.estimate_cost(
            "claude-sonnet-4-6-20260101",
            input_tokens=1000,
            output_tokens=500,
            is_batch=True,
        )
        expected = 1.50 * 1000 / 1_000_000 + 7.50 * 500 / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_batch_fallback_to_normal_pricing(self):
        # gpt-4-turbo has no batch pricing (None) — should fall back to normal rates
        cost = pricing.estimate_cost(
            "gpt-4-turbo-2024-04-09",
            input_tokens=1000,
            output_tokens=500,
            is_batch=True,
        )
        expected = 10.0 * 1000 / 1_000_000 + 30.0 * 500 / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10


class TestLongContextPricing:
    def test_standard_below_threshold(self):
        # 100K tokens is below 200K threshold — use standard rates
        cost = pricing.estimate_cost(
            "claude-opus-4-5-20260101", input_tokens=100_000, output_tokens=1000
        )
        expected = 5.0 * 100_000 / 1_000_000 + 25.0 * 1000 / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_long_context_above_threshold(self):
        # 300K total_input > 200K threshold — use long-context rates
        cost = pricing.estimate_cost(
            "claude-opus-4-5-20260101", input_tokens=300_000, output_tokens=1000
        )
        expected = 10.0 * 300_000 / 1_000_000 + 37.50 * 1000 / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_cache_tokens_count_toward_threshold(self):
        # 100K input + 60K cache_write + 50K cache_read = 210K > 200K
        cost = pricing.estimate_cost(
            "claude-sonnet-4-5-20260101",
            input_tokens=100_000,
            output_tokens=1000,
            cache_write_tokens=60_000,
            cache_read_tokens=50_000,
        )
        # Long context rates: input=6.0, output=22.50
        expected = (
            6.0 * 100_000 / 1_000_000
            + 22.50 * 1000 / 1_000_000
            + 3.75 * 60_000 / 1_000_000
            + 0.30 * 50_000 / 1_000_000
        )
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_exact_threshold_uses_standard_rates(self):
        # total_input == threshold (not strictly greater) → standard rates
        cost = pricing.estimate_cost(
            "claude-opus-4-5-20260101", input_tokens=200_000, output_tokens=1000
        )
        expected = 5.0 * 200_000 / 1_000_000 + 25.0 * 1000 / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_one_above_threshold_uses_long_context(self):
        # total_input == threshold + 1 → long-context rates
        cost = pricing.estimate_cost(
            "claude-opus-4-5-20260101", input_tokens=200_001, output_tokens=1000
        )
        expected = 10.0 * 200_001 / 1_000_000 + 37.50 * 1000 / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_no_long_context_for_models_without_threshold(self):
        # claude-haiku-4-5 has no long_context_threshold — always standard
        p = pricing.get("claude-haiku-4-5-20251001")
        assert p is not None
        assert p.long_context_threshold is None

    def test_model_pricing_fields(self):
        p = pricing.get("claude-opus-4-5-20260101")
        assert p is not None
        assert p.long_context_threshold == 200_000
        assert p.long_context_input == 10.0
        assert p.long_context_output == 37.50


class TestFastModePricing:
    def test_fast_mode(self):
        cost = pricing.estimate_cost(
            "claude-opus-5",
            input_tokens=1000,
            output_tokens=500,
            is_fast=True,
        )
        expected = 10.0 * 1000 / 1_000_000 + 50.0 * 500 / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_fast_mode_takes_priority_over_batch(self):
        cost = pricing.estimate_cost(
            "claude-opus-5",
            input_tokens=1000,
            output_tokens=500,
            is_fast=True,
            is_batch=True,
        )
        expected = 10.0 * 1000 / 1_000_000 + 50.0 * 500 / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_fast_mode_takes_priority_over_long_context(self):
        # No bundled model carries both fast and long-context rates today, so
        # exercise the precedence with a registered synthetic entry.
        reg = PricingRegistry()
        reg.register(
            "fast-long",
            ModelPricing(
                input=1.0,
                output=2.0,
                fast_input=5.0,
                fast_output=10.0,
                long_context_threshold=200_000,
                long_context_input=2.0,
                long_context_output=4.0,
            ),
        )
        cost = reg.estimate_cost(
            "fast-long-v1", input_tokens=300_000, output_tokens=1000, is_fast=True
        )
        # Should use fast rates, not long-context
        expected = 5.0 * 300_000 / 1_000_000 + 10.0 * 1000 / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_fast_fallback_when_no_fast_pricing(self):
        # gpt-4o has no fast pricing — should use standard
        cost = pricing.estimate_cost(
            "gpt-4o-2024-08-06",
            input_tokens=1000,
            output_tokens=500,
            is_fast=True,
        )
        expected = 2.50 * 1000 / 1_000_000 + 10.0 * 500 / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_fast_fallback_to_batch_when_both_flags_set(self):
        # No fast pricing → falls through to batch
        cost = pricing.estimate_cost(
            "claude-sonnet-4-6-20260101",
            input_tokens=1000,
            output_tokens=500,
            is_fast=True,
            is_batch=True,
        )
        expected = 1.50 * 1000 / 1_000_000 + 7.50 * 500 / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_model_pricing_fast_fields(self):
        p = pricing.get("claude-opus-5")
        assert p is not None
        assert p.fast_input == 10.0
        assert p.fast_output == 50.0


class TestConvenienceEstimateCost:
    def test_returns_float_for_known(self):
        cost = estimate_cost("claude-sonnet-4-6-20260101", input_tokens=1000, output_tokens=500)
        assert isinstance(cost, float)
        assert cost > 0

    def test_unknown_returns_none(self):
        assert estimate_cost("unknown-model", input_tokens=1000) is None

    def test_is_fast_passthrough(self):
        cost = estimate_cost("claude-opus-5", input_tokens=1000, output_tokens=500, is_fast=True)
        assert cost is not None
        expected = 10.0 * 1000 / 1_000_000 + 50.0 * 500 / 1_000_000
        assert abs(cost - expected) < 1e-10


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
        assert reg.has("claude-sonnet-4-6-20260101")

    def test_load_long_context_fields(self, tmp_path: Path):
        toml_content = (
            "[my-model]\n"
            "input = 1.0\n"
            "output = 2.0\n"
            "long_context_threshold = 100_000\n"
            "long_context_input = 3.0\n"
            "long_context_output = 6.0\n"
        )
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(toml_content)

        reg = PricingRegistry()
        reg.load(toml_file)
        p = reg.get("my-model-v1")
        assert p is not None
        assert p.long_context_threshold == 100_000
        assert p.long_context_input == 3.0
        assert p.long_context_output == 6.0

    def test_load_fast_fields(self, tmp_path: Path):
        toml_content = (
            "[my-model]\ninput = 1.0\noutput = 2.0\nfast_input = 5.0\nfast_output = 10.0\n"
        )
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(toml_content)

        reg = PricingRegistry()
        reg.load(toml_file)
        p = reg.get("my-model-v1")
        assert p is not None
        assert p.fast_input == 5.0
        assert p.fast_output == 10.0


class TestModelPricingNone:
    def test_none_semantics(self):
        # Models without cache/batch have None, not 0.0
        p = pricing.get("gpt-4-turbo-2024-04-09")
        assert p is not None
        assert p.cache_write is None
        assert p.cache_read is None
        assert p.batch_input is None
        assert p.batch_output is None
        assert p.long_context_threshold is None
        assert p.fast_input is None

    def test_claude_has_cache_pricing(self):
        p = pricing.get("claude-sonnet-4-6-20260101")
        assert p is not None
        assert p.cache_write is not None
        assert p.cache_read is not None


class TestPricingCache:
    """get() memoizes longest-prefix lookups and invalidates on every mutation (perf review #4)."""

    def test_cache_returns_consistent_results(self):
        reg = PricingRegistry()
        assert reg.get("claude-sonnet-4-6-20260101") is reg.get("claude-sonnet-4-6-20260101")

    def test_cache_invalidates_on_register_and_unregister(self):
        reg = PricingRegistry()
        model = "brand-new-model-xyz"
        assert reg.get(model) is None  # caches the None (known-unpriced)
        reg.register(model, ModelPricing(input=1.0, output=2.0))
        assert reg.get(model) is not None  # cache cleared -> reflects the registration
        reg.unregister(model)
        assert reg.get(model) is None  # cleared again

    def test_cache_invalidates_on_reset(self):
        reg = PricingRegistry()
        reg.register("temp-model-abc", ModelPricing(input=1.0, output=2.0))
        assert reg.get("temp-model-abc") is not None
        reg.reset()
        assert reg.get("temp-model-abc") is None  # reset discarded it and cleared the cache
