"""Tests for _pricing.py — PricingRegistry."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from ai_arch_toolkit.core import ModelPricing, PricingRegistry, _pricing, pricing
from ai_arch_toolkit.core._pricing import estimate_cost


class TestPricingRegistryDefaults:
    """The module-level singleton loads from _default_pricing.toml."""

    def test_has_claude_models(self):
        assert pricing.has("claude-opus-5")
        assert pricing.has("claude-sonnet-5")
        assert pricing.has("claude-fable-5")
        assert pricing.has("claude-opus-4-8")
        assert not pricing.has("claude-opus-4-1")

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
        assert pricing.has("gpt-5.6")
        assert pricing.has("gpt-5.6-cyber")
        assert pricing.has("gpt-daybreak-blue-latest")
        assert pricing.has("gpt-daybreak-red-latest")
        assert pricing.has("chat-latest")

    def test_has_gemini_models(self):
        assert pricing.has("gemini-3.7-flash")
        assert pricing.has("gemini-3.6-flash")
        assert pricing.has("gemini-3.5-flash")
        assert pricing.has("gemini-3.5-flash-lite")
        assert pricing.has("gemini-3.1-pro-preview-20260301")
        assert pricing.has("gemini-3-flash-preview")
        assert pricing.has("gemini-2.5-flash")
        assert pricing.has("gemini-2.5-pro")
        assert not pricing.has("gemini-2.0-flash-lite")

    def test_has_o_series_models(self):
        assert pricing.has("o3")
        assert pricing.has("o4-mini-deep-research")
        assert pricing.has("grok-3")

    def test_has_grok4_models(self):
        assert pricing.has("grok-4.5")
        assert pricing.has("grok-4.20-0309-reasoning")
        assert pricing.has("grok-build-0.1")
        assert pricing.has("grok-4")
        assert pricing.has("grok-4-1-fast-reasoning")
        assert pricing.has("grok-4-fast-reasoning")
        assert pricing.has("grok-code-fast")
        assert pricing.has("grok-code-fast-1")
        assert pricing.has("grok-build-latest")

    def test_grok_47_pricing(self):
        # https://docs.x.ai/developers/models/grok-4.7 (2026-09-25): the whole request is billed
        # at the long-context rates once the prompt reaches 200k tokens; no Batch API.
        entry = pricing.get("grok-4.7")
        assert entry is not None
        assert (entry.input, entry.output, entry.cache_read) == (2.0, 6.0, 0.50)
        assert entry.long_context_threshold == 200_000
        assert entry.long_context_inclusive
        assert (entry.long_context_input, entry.long_context_output) == (4.0, 12.0)
        assert entry.long_context_cache_read == 1.0
        assert entry.batch_input is None

    def test_grok_46_has_its_own_entry_not_the_grok_4_prefix(self):
        entry = pricing.get("grok-4.6")
        assert entry is not None
        assert entry.input == 2.0
        assert entry.output == 6.0
        assert entry.cache_read == 0.50
        assert entry != pricing.get("grok-4")

    def test_muse_spark_tiers(self):
        standard = pricing.get("muse-spark-1.3")
        assert standard is not None
        assert (standard.input, standard.output, standard.cache_read) == (1.25, 4.25, 0.15)
        assert pricing.get("muse-spark-1.2") == standard
        assert pricing.get("muse-spark-1.1") == standard
        # Without entries of its own, the contributor tier would match the standard prefix.
        contributor = pricing.get("muse-spark-1.3-contributor")
        assert contributor is not None
        assert (contributor.input, contributor.output, contributor.cache_read) == (
            0.10,
            0.20,
            0.002,
        )
        assert pricing.get("muse-spark-1.2-contributor") == contributor

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
        assert p.input == 4.0  # gpt-5.6-sol, not gpt-5's $1.25

    def test_gpt56_default_alias_has_sol_pricing(self):
        p = pricing.get("gpt-5.6")
        assert p is not None
        assert p.input == 4.0
        assert p.output == 20.0
        assert p.cache_write == 5.0

    def test_current_aliases_share_the_canonical_prices(self):
        aliases = {
            "gpt-5.6": "gpt-5.6-sol",
            "gpt-daybreak-blue-latest": "gpt-5.6-sol",
            "gpt-daybreak-red-latest": "gpt-5.6-cyber",
            "grok-code-fast": "grok-build-0.1",
            "grok-code-fast-1": "grok-build-0.1",
            "grok-build-latest": "grok-4.5",
        }
        for alias, canonical in aliases.items():
            assert pricing.get(alias) == pricing.get(canonical)

    def test_gpt56_tier_prices(self):
        expected = {
            "gpt-5.6-sol": (4.0, 20.0, 0.40),
            "gpt-5.6-terra": (2.0, 12.0, 0.20),
            "gpt-5.6-luna": (0.20, 1.20, 0.020),
            "gpt-5.6-cyber": (12.50, 75.0, 1.25),
        }
        for model, rates in expected.items():
            p = pricing.get(model)
            assert p is not None
            assert (p.input, p.output, p.cache_read) == rates

    def test_gpt6_sol_and_luna_prices(self):
        # https://developers.openai.com/api/docs/models/gpt-6-sol and .../gpt-6-luna (2026-09-25)
        expected = {
            # (input, cache_read, cache_write, output), then above 272K, batch, and fast
            "gpt-6-sol": ((2.0, 0.20, 2.50, 10.0), (4.0, 15.0), (1.0, 5.0), (4.0, 20.0)),
            "gpt-6-luna": ((0.10, 0.010, 0.125, 0.50), (0.20, 0.75), (0.05, 0.25), (0.20, 1.0)),
        }
        for model, (standard, long, batch, fast) in expected.items():
            p = pricing.get(model)
            assert p is not None
            assert (p.input, p.cache_read, p.cache_write, p.output) == standard
            assert p.long_context_threshold == 272_000
            assert not p.long_context_inclusive  # "more than 272K input tokens"
            assert (p.long_context_input, p.long_context_output) == long
            assert (p.batch_input, p.batch_output) == batch
            assert (p.fast_input, p.fast_output) == fast

    def test_gpt61_sol_prices(self):
        # https://developers.openai.com/api/docs/models/gpt-6.1-sol (2026-10-01): cached input at
        # 5% of input (GPT-6 Sol's is 10%), cache writes at 1.25x; above 272K input tokens 2x input
        # and cache rates and 1.5x output; Batch and Flex 50% of Standard; Fast mode 2x.
        p = pricing.get("gpt-6.1-sol")
        assert p is not None
        assert (p.input, p.cache_read, p.cache_write, p.output) == (2.0, 0.10, 2.50, 10.0)
        assert p.long_context_threshold == 272_000
        assert not p.long_context_inclusive
        long = (
            p.long_context_input,
            p.long_context_cache_read,
            p.long_context_cache_write,
            p.long_context_output,
        )
        assert long == (4.0, 0.20, 5.0, 15.0)
        batch = (p.batch_input, p.batch_cache_read, p.batch_cache_write, p.batch_output)
        assert batch == (1.0, 0.05, 1.25, 5.0)
        fast = (p.fast_input, p.fast_cache_read, p.fast_cache_write, p.fast_output)
        assert fast == (4.0, 0.20, 5.0, 20.0)
        fast_long = (
            p.fast_long_context_input,
            p.fast_long_context_cache_read,
            p.fast_long_context_cache_write,
            p.fast_long_context_output,
        )
        assert fast_long == (8.0, 0.40, 10.0, 30.0)

    def test_current_anthropic_pricing(self):
        sonnet = pricing.get("claude-sonnet-5")
        opus_48 = pricing.get("claude-opus-4-8")
        assert sonnet is not None
        assert opus_48 is not None
        assert (sonnet.input, sonnet.output, sonnet.cache_read) == (2.0, 10.0, 0.20)
        assert (opus_48.fast_input, opus_48.fast_output) == (10.0, 50.0)

    def test_opus_55_pricing(self):
        # https://platform.claude.com/docs/en/about-claude/pricing (2026-09-25): cache reads at
        # 0.05x input, and fast mode at 2x with the caching multipliers on top.
        p = pricing.get("claude-opus-5-5")
        assert p is not None
        assert (p.input, p.output, p.cache_write, p.cache_read) == (4.0, 20.0, 5.0, 0.20)
        assert (p.batch_input, p.batch_output, p.batch_cache_read) == (2.0, 10.0, 0.10)
        assert (p.fast_input, p.fast_output, p.fast_cache_read) == (8.0, 40.0, 0.40)

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

    def test_gemini_37_promotional_pricing(self):
        p = pricing.get("gemini-3.7-flash")
        assert p is not None
        assert p.input == 0.75
        assert p.output == 3.75
        assert p.batch_input == 0.375

    def test_grok_code_fast(self):
        p = pricing.get("grok-code-fast-1")
        assert p is not None
        assert p.input == 1.0
        assert p.output == 2.0
        assert p.cache_read == 0.20

    def test_xai_batch_support_and_discount(self):
        grok_43 = pricing.get("grok-4.3")
        grok_45 = pricing.get("grok-4.5")
        build = pricing.get("grok-build-0.1")
        assert grok_43 is not None
        assert grok_45 is not None
        assert build is not None
        assert (grok_43.batch_input, grok_43.batch_output) == (1.0, 2.0)
        assert grok_43.batch_cache_read == 0.16
        assert grok_45.batch_input is None
        assert build.batch_input is None


class TestPricingRegistryRegister:
    def test_register_is_exact_and_covers_dated_snapshots(self):
        reg = PricingRegistry()
        reg.register("my-model", ModelPricing(input=1.0, output=2.0))
        assert reg.get("my-model") == ModelPricing(input=1.0, output=2.0)
        assert reg.get("my-model-20260101") == ModelPricing(input=1.0, output=2.0)
        assert reg.get("my-model-v1") is None  # a variant is another model, never inherited

    def test_override_existing(self):
        reg = PricingRegistry()
        reg.register("claude-sonnet-4-6", ModelPricing(input=99.0, output=99.0))
        p = reg.get("claude-sonnet-4-6-20260101")
        assert p is not None
        assert p.input == 99.0

    def test_prefix_registration_is_explicit_for_local_families(self):
        reg = PricingRegistry()
        reg.register("llama3", ModelPricing(), match="prefix")
        assert reg.get("llama3.2:8b") == ModelPricing()
        assert reg.estimate_cost("llama3.1", input_tokens=1000, output_tokens=10) == 0.0
        assert reg.get("llama2") is None

    def test_an_exact_entry_wins_over_a_prefix(self):
        reg = PricingRegistry()
        reg.register("acme-", ModelPricing(input=1.0), match="prefix")
        reg.register("acme-large", ModelPricing(input=9.0))
        assert reg.get("acme-large") == ModelPricing(input=9.0)
        assert reg.get("acme-small") == ModelPricing(input=1.0)

    def test_unregister_removes_either_kind(self):
        reg = PricingRegistry()
        reg.register("temp-model", ModelPricing(input=1.0, output=1.0))
        reg.register("temp-", ModelPricing(input=2.0), match="prefix")
        assert reg.get("temp-model") == ModelPricing(input=1.0, output=1.0)
        reg.unregister("temp-model")
        assert reg.get("temp-model") == ModelPricing(input=2.0)
        reg.unregister("temp-")
        assert not reg.has("temp-model")


class TestPricingRegistryReset:
    def test_reset_clears_custom(self):
        reg = PricingRegistry()
        reg.register("custom", ModelPricing(input=1.0, output=1.0))
        reg.register("custom-", ModelPricing(input=1.0, output=1.0), match="prefix")
        reg.reset()
        assert not reg.has("custom")
        assert not reg.has("custom-v1")
        # Defaults still there
        assert reg.has("claude-sonnet-4-6-20260101")


class TestIdGrammar:
    """Prices resolve by an id's own entry or a dated snapshot of one (costura D, R6)."""

    @pytest.mark.parametrize(
        "model",
        [
            "o3-pro",
            "o3-deep-research",
            "gpt-4o-audio-preview",
            "gpt-5-codex",
            "gemini-2.5-flash-image",
            "grok-4.6-mini",
        ],
    )
    def test_variants_of_priced_ids_have_no_price(self, model: str):
        assert pricing.get(model) is None

    def test_a_snapshot_with_its_own_tariff_keeps_it(self):
        # https://developers.openai.com/api/docs/pricing (2026-09-18): the May 2024 gpt-4o
        # snapshot costs more than the id it is a snapshot of; the other snapshots do not.
        dated = pricing.get("gpt-4o-2024-05-13")
        assert dated is not None
        assert (dated.input, dated.output) == (5.0, 15.0)
        assert pricing.get("gpt-4o-2024-08-06") == pricing.get("gpt-4o")

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4.1-nano",
            "gpt-4.1-nano-2025-04-14",
            "o1",
            "o1-2024-12-17",
            "o1-pro",
            "o3-mini",
            "o4-mini",
            "o4-mini-2025-04-16",
        ],
    )
    def test_models_openai_shuts_down_have_no_price(self, model: str):
        # Shut down by 2026-10-23 (https://developers.openai.com/api/docs/deprecations).
        assert pricing.get(model) is None

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("grok-4.20-reasoning", "grok-4.20"),
            ("grok-4.20-non-reasoning", "grok-4.20"),
            ("grok-4.20-0309-reasoning", "grok-4.20"),
            ("grok-4.20-0309-non-reasoning", "grok-4.20"),
            ("grok-4.20-multi-agent", "grok-4.20"),
            ("grok-4.20-multi-agent-0309", "grok-4.20"),
            ("grok-4-fast-reasoning", "grok-4-fast"),
            ("grok-4-fast-non-reasoning", "grok-4-fast"),
            ("grok-4-1-fast-reasoning", "grok-4-1-fast"),
            ("grok-4-1-fast-non-reasoning", "grok-4-1-fast"),
            ("grok-3-fast", "grok-3"),
            ("grok-3-mini-fast", "grok-3-mini"),
            ("gemini-3-pro-preview", "gemini-3-pro"),
            ("gemini-3.1-pro-preview", "gemini-3.1-pro"),
            ("gemini-3-flash-preview", "gemini-3-flash"),
            ("gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite"),
            ("gpt-5.1", "gpt-5"),
        ],
    )
    def test_documented_ids_share_their_entry(self, alias: str, canonical: str):
        assert pricing.get(canonical) is not None
        assert pricing.get(alias) == pricing.get(canonical)

    def test_the_new_cyber_entry(self):
        cyber = pricing.get("gpt-5.5-cyber")
        assert cyber is not None
        assert (cyber.input, cyber.output, cyber.cache_read) == (12.50, 75.0, 1.25)


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
        # gpt-5.5-pro has no cache pricing (None) — cache tokens should not contribute
        cost = pricing.estimate_cost(
            "gpt-5.5-pro",
            input_tokens=1000,
            output_tokens=0,
            cache_write_tokens=500,
        )
        expected = 30.0 * 1000 / 1_000_000
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
        # Meta has no batch pricing (None) — should fall back to normal rates
        cost = pricing.estimate_cost(
            "muse-spark-1.3",
            input_tokens=1000,
            output_tokens=500,
            is_batch=True,
        )
        expected = 1.25 * 1000 / 1_000_000 + 4.25 * 500 / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_batch_cache_prices(self):
        cost = pricing.estimate_cost(
            "gemini-3.7-flash",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=1000,
            is_batch=True,
        )
        expected = (0.375 * 1000 + 1.875 * 500 + 0.0375 * 1000) / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_legacy_custom_cache_rates_remain_the_mode_fallback(self):
        reg = PricingRegistry()
        reg.register(
            "legacy-custom",
            ModelPricing(input=1.0, output=2.0, cache_read=0.1, batch_input=0.5),
        )
        cost = reg.estimate_cost(
            "legacy-custom", input_tokens=1000, cache_read_tokens=1000, is_batch=True
        )
        assert cost is not None
        assert abs(cost - 0.0006) < 1e-10

    def test_explicit_zero_cache_rate_does_not_fall_back(self):
        reg = PricingRegistry()
        reg.register(
            "free-batch-cache",
            ModelPricing(
                input=1.0,
                output=2.0,
                cache_read=0.1,
                batch_input=0.5,
                batch_cache_read=0.0,
            ),
        )
        cost = reg.estimate_cost(
            "free-batch-cache", input_tokens=1000, cache_read_tokens=1000, is_batch=True
        )
        assert cost is not None
        assert abs(cost - 0.0005) < 1e-10


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
        # Long context rates also apply to cached input.
        expected = (
            6.0 * 100_000 / 1_000_000
            + 22.50 * 1000 / 1_000_000
            + 7.50 * 60_000 / 1_000_000
            + 0.60 * 50_000 / 1_000_000
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

    def test_inclusive_xai_threshold_uses_long_context_at_200k(self):
        cost = pricing.estimate_cost(
            "grok-4.6", input_tokens=199_000, cache_read_tokens=1000, output_tokens=1000
        )
        expected = 4.0 * 199_000 / 1_000_000 + 1.0 * 1000 / 1_000_000 + 12.0 / 1000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_batch_long_context_uses_combined_rates(self):
        cost = pricing.estimate_cost(
            "gpt-5.6-sol",
            input_tokens=272_001,
            output_tokens=1000,
            cache_write_tokens=1000,
            cache_read_tokens=1000,
            is_batch=True,
        )
        expected = (
            4.0 * 272_001 / 1_000_000
            + 15.0 * 1000 / 1_000_000
            + 5.0 * 1000 / 1_000_000
            + 0.40 * 1000 / 1_000_000
        )
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
        # Legacy custom tables without combined rates retain fast-mode precedence.
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
            "fast-long", input_tokens=300_000, output_tokens=1000, is_fast=True
        )
        # Should use fast rates, not long-context
        expected = 5.0 * 300_000 / 1_000_000 + 10.0 * 1000 / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_codex_fast_price_includes_fast_cache_rate(self):
        cost = pricing.estimate_cost(
            "gpt-5.3-codex",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=1000,
            is_fast=True,
        )
        expected = (3.50 * 1000 + 28.0 * 500 + 0.35 * 1000) / 1_000_000
        assert cost is not None
        assert abs(cost - expected) < 1e-10

    def test_fast_long_context_and_cache_rates(self):
        cost = pricing.estimate_cost(
            "gpt-5.6-sol",
            input_tokens=272_001,
            output_tokens=1000,
            cache_write_tokens=1000,
            cache_read_tokens=1000,
            is_fast=True,
        )
        expected = (
            16.0 * 272_001 / 1_000_000
            + 60.0 * 1000 / 1_000_000
            + 20.0 * 1000 / 1_000_000
            + 1.60 * 1000 / 1_000_000
        )
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
        p = reg.get("my-custom-model")
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
        assert reg.has("custom")
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
        p = reg.get("my-model")
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
        p = reg.get("my-model")
        assert p is not None
        assert p.fast_input == 5.0
        assert p.fast_output == 10.0

    def test_load_mode_specific_cache_and_long_context_fields(self, tmp_path: Path):
        toml_content = (
            "[my-model]\n"
            "input = 1.0\n"
            "output = 2.0\n"
            "batch_input = 0.5\n"
            "batch_cache_read = 0.1\n"
            "long_context_threshold = 100\n"
            "long_context_inclusive = true\n"
            "batch_long_context_input = 1.0\n"
            "batch_long_context_cache_read = 0.2\n"
            "fast_input = 3.0\n"
            "fast_cache_read = 0.3\n"
        )
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(toml_content)

        reg = PricingRegistry()
        reg.load(toml_file)
        p = reg.get("my-model")
        assert p is not None
        assert p.batch_cache_read == 0.1
        assert p.long_context_inclusive is True
        assert p.batch_long_context_input == 1.0
        assert p.batch_long_context_cache_read == 0.2
        assert p.fast_cache_read == 0.3

    def test_entries_take_aliases_and_an_explicit_prefix(self, tmp_path: Path):
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            '["acme-1"]\ninput = 1.0\noutput = 2.0\naliases = ["acme-1-fast"]\n\n'
            '["local-"]\nmatch = "prefix"\n'
        )
        reg = PricingRegistry()
        reg.load(toml_file)
        assert reg.get("acme-1-fast") == ModelPricing(input=1.0, output=2.0)
        assert reg.get("acme-1-mini") is None
        assert reg.get("local-llama") == ModelPricing()

    def test_an_unknown_match_kind_is_refused(self, tmp_path: Path):
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text('["acme-"]\nmatch = "suffix"\n')
        with pytest.raises(ValueError, match="match"):
            PricingRegistry().load(toml_file)
        kind: Any = "suffix"
        with pytest.raises(ValueError, match="match"):
            PricingRegistry().register("acme-", ModelPricing(), match=kind)


class TestModelPricingNone:
    def test_none_semantics(self, tmp_path: Path):
        # Models without cache/batch have None, not 0.0
        table = tmp_path / "prices.toml"
        table.write_text("[acme-1]\ninput = 1.0\noutput = 2.0\n")
        registry = PricingRegistry()
        registry.load(table)
        p = registry.get("acme-1-2026-01-01")
        assert p is not None
        assert p.cache_write is None
        assert p.cache_read is None
        assert p.batch_input is None
        assert p.batch_output is None
        assert p.batch_cache_read is None
        assert p.long_context_threshold is None
        assert p.long_context_inclusive is False
        assert p.fast_input is None

    def test_claude_has_cache_pricing(self):
        p = pricing.get("claude-sonnet-4-6-20260101")
        assert p is not None
        assert p.cache_write is not None
        assert p.cache_read is not None


class TestPricingCache:
    """get() memoizes lookups and invalidates on every mutation (perf review #4)."""

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


@pytest.mark.parametrize(
    "is_batch,is_fast,multiplier", [(False, False, 1), (True, False, 0.5), (False, True, 2)]
)
@pytest.mark.parametrize("tokens", [272_000, 272_001])
def test_astra_cost_tiers(is_batch, is_fast, multiplier, tokens):
    long_context = tokens > 272_000
    input_rate = 20 if long_context else 10
    output_rate = 75 if long_context else 50
    cost = pricing.estimate_cost(
        "gpt-6-astra",
        input_tokens=tokens - 2000,
        cache_read_tokens=1000,
        cache_write_tokens=1000,
        output_tokens=1000,
        is_batch=is_batch,
        is_fast=is_fast,
    )
    expected = (
        multiplier
        * (
            input_rate * (tokens - 2000)
            + input_rate * 0.1 * 1000
            + input_rate * 1.25 * 1000
            + output_rate * 1000
        )
        / 1_000_000
    )
    assert cost == pytest.approx(expected)


@pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-mythos-5-1"])
@pytest.mark.parametrize("batch,expected", [(False, 60.25), (True, 30.125)])
def test_claude_51_prices_cache_at_its_own_rate(model, batch, expected):
    cost = pricing.estimate_cost(
        model,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        is_batch=batch,
    )
    assert cost == pytest.approx(expected)
    assert model in pricing.list_models()


@pytest.mark.parametrize("batch,expected", [(False, 4.575), (True, 2.2875)])
def test_gemini_38_has_its_own_promotional_price(batch, expected):
    cost = pricing.estimate_cost(
        "gemini-3.8-flash",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        is_batch=batch,
    )
    assert cost == pytest.approx(expected)
    assert "gemini-3.8-flash" in pricing.list_models()


class TestImagePricing:
    """Image tokens and images, priced apart from text (I02)."""

    @staticmethod
    def registry(**rates: float) -> PricingRegistry:
        registry = PricingRegistry()
        registry.register("img", ModelPricing(**rates))
        return registry

    def test_image_tokens_take_the_image_rates(self) -> None:
        registry = self.registry(input=5.0, output=10.0, image_input=8.0, image_output=30.0)
        cost = registry.estimate_cost(
            "img",
            input_tokens=21,
            output_tokens=4,
            image_input_tokens=1024,
            image_output_tokens=196,
        )
        assert cost == pytest.approx((21 * 5 + 4 * 10 + 1024 * 8 + 196 * 30) / 1e6)

    def test_an_image_rate_left_out_falls_back_to_the_text_rate(self) -> None:
        registry = self.registry(input=0.5, output=3.0)
        cost = registry.estimate_cost("img", image_input_tokens=1000, image_output_tokens=1000)
        assert cost == pytest.approx((1000 * 0.5 + 1000 * 3.0) / 1e6)

    def test_batch_takes_the_batch_image_rates_or_the_standard_image_rates(self) -> None:
        registry = self.registry(
            input=5.0,
            output=10.0,
            image_output=30.0,
            batch_input=2.5,
            batch_output=5.0,
            batch_image_output=15.0,
        )
        assert registry.estimate_cost(
            "img", image_output_tokens=1_000_000, is_batch=True
        ) == pytest.approx(15.0)
        partial = self.registry(
            input=5.0, output=10.0, image_input=8.0, batch_input=2.5, batch_output=5.0
        )
        assert partial.estimate_cost(
            "img", image_input_tokens=1_000_000, is_batch=True
        ) == pytest.approx(8.0)

    def test_per_image_charges_each_image(self) -> None:
        registry = self.registry(per_image=0.01, batch_input=0.0, batch_per_image=0.005)
        assert registry.estimate_cost("img", image_count=3) == pytest.approx(0.03)
        assert registry.estimate_cost("img", image_count=3, is_batch=True) == pytest.approx(0.015)

    def test_the_module_wrapper_passes_the_image_counters(self) -> None:
        pricing.register("img-wrapper", ModelPricing(image_output=30.0, per_image=0.01))
        try:
            cost = estimate_cost("img-wrapper", image_output_tokens=1_000_000, image_count=1)
            assert cost == pytest.approx(30.01)
        finally:
            pricing.unregister("img-wrapper")

    def test_image_rates_load_from_toml(self, tmp_path: Path) -> None:
        table = tmp_path / "prices.toml"
        table.write_text("[img]\ninput = 5.0\nimage_output = 30.0\nper_image = 0.02\n")
        registry = PricingRegistry()
        registry.load(table)
        price = registry.get("img")
        assert price is not None
        assert (price.image_output, price.per_image) == (30.0, 0.02)


class TestDatedPrices:
    """A promotional price has its last day and the price after it (D50)."""

    PROMO = ModelPricing(
        input=4.0,
        output=20.0,
        until=date(2026, 11, 21),
        then=ModelPricing(input=5.0, output=30.0),
    )

    def test_the_price_of_a_day_is_the_promotion_until_its_last_day(self) -> None:
        assert self.PROMO.on(date(2026, 11, 21)).input == 4.0
        assert self.PROMO.on(date(2026, 11, 22)) == ModelPricing(input=5.0, output=30.0)
        assert ModelPricing(input=1.0).on(date(2030, 1, 1)).input == 1.0

    def test_the_registry_reads_the_price_of_today_or_of_a_day_asked(self, monkeypatch) -> None:
        registry = PricingRegistry()
        registry.register("promo", self.PROMO)
        assert registry.get("promo", on=date(2026, 12, 1)).input == 5.0  # type: ignore[union-attr]
        monkeypatch.setattr(_pricing, "_today", lambda: date(2026, 11, 1))
        assert registry.estimate_cost("promo", input_tokens=1_000_000) == pytest.approx(4.0)
        monkeypatch.setattr(_pricing, "_today", lambda: date(2026, 11, 22))
        assert registry.estimate_cost("promo", input_tokens=1_000_000) == pytest.approx(5.0)

    def test_the_meter_prices_by_the_day(self, monkeypatch) -> None:
        from ai_arch_toolkit.core import OperationRequest, Usage

        registry = PricingRegistry()
        registry.register("promo", self.PROMO)
        request = OperationRequest(kind="llm", parent_span_id="run", model="promo")
        usage = Usage(output_tokens=1_000_000)
        monkeypatch.setattr(_pricing, "_today", lambda: date(2026, 11, 21))
        assert registry.price(request, usage).amount.to_float() == pytest.approx(20.0)
        monkeypatch.setattr(_pricing, "_today", lambda: date(2026, 11, 22))
        assert registry.price(request, usage).amount.to_float() == pytest.approx(30.0)

    def test_a_dated_price_loads_from_toml(self, tmp_path: Path) -> None:
        table = tmp_path / "prices.toml"
        table.write_text(
            '["promo-1.0"]\naliases = ["promo-latest"]\ninput = 4.0\noutput = 20.0\n'
            "until = 2026-11-21\n\n"
            '["promo-1.0".then]\ninput = 5.0\noutput = 30.0\n'
        )
        registry = PricingRegistry()
        registry.load(table)
        for model in ("promo-1.0", "promo-latest"):
            assert registry.get(model, on=date(2026, 11, 21)).input == 4.0  # type: ignore[union-attr]
            assert registry.get(model, on=date(2026, 11, 22)).output == 30.0  # type: ignore[union-attr]

    def test_an_until_that_is_not_a_date_is_refused(self, tmp_path: Path) -> None:
        table = tmp_path / "prices.toml"
        table.write_text('[promo]\ninput = 4.0\nuntil = "2026-11-21"\n')
        with pytest.raises(ValueError, match="until must be a TOML date"):
            PricingRegistry().load(table)


class TestTablePromotionsAndVendors:
    """The table's dated promotions and the vendors the ai-network uses (G-20)."""

    @pytest.mark.parametrize(
        ("model", "last_day", "now", "after"),
        [
            ("gpt-5.6-sol", date(2026, 11, 21), (4.0, 20.0, 0.40), (5.0, 30.0, 0.50)),
            ("gpt-5.6", date(2026, 11, 21), (4.0, 20.0, 0.40), (5.0, 30.0, 0.50)),
            ("gemini-3.8-flash", date(2026, 12, 31), (0.75, 3.75, 0.075), (1.50, 7.50, 0.15)),
            ("gemini-3.7-flash", date(2026, 12, 31), (0.75, 3.75, 0.075), (1.50, 7.50, 0.15)),
            ("gemini-3.6-flash", date(2026, 12, 31), (0.75, 3.75, 0.075), (1.50, 7.50, 0.15)),
        ],
    )
    def test_a_promotion_ends_on_its_day(self, model, last_day, now, after) -> None:
        on_last_day = pricing.get(model, on=last_day)
        on_next_day = pricing.get(model, on=last_day + timedelta(days=1))
        assert on_last_day is not None and on_next_day is not None
        assert on_last_day.until == last_day
        assert (on_last_day.input, on_last_day.output, on_last_day.cache_read) == now
        assert (on_next_day.input, on_next_day.output, on_next_day.cache_read) == after

    def test_gpt_5_6_sol_long_context_and_batch_follow_the_promotion(self) -> None:
        after = pricing.get("gpt-5.6-sol", on=date(2026, 11, 22))
        assert after is not None
        assert (after.long_context_input, after.long_context_output) == (10.0, 45.0)
        assert (after.batch_input, after.batch_output) == (2.50, 15.0)

    @pytest.mark.parametrize(
        ("model", "rates"),
        [
            ("deepseek-flash", (0.30, 1.20, 0.006)),
            ("deepseek-v4-flash", (0.30, 1.20, 0.006)),
            ("deepseek-v4-pro", (1.32, 3.96, 0.044)),
            ("mistral-small-2603", (0.15, 0.60, 0.015)),
            ("mistral-small-latest", (0.15, 0.60, 0.015)),
            ("codestral-2508", (0.30, 0.90, 0.03)),
            ("codestral-latest", (0.30, 0.90, 0.03)),
            ("poolside/laguna-s-2.1", (0.10, 0.20, 0.01)),
        ],
    )
    def test_the_vendors_the_ai_network_uses_are_priced(self, model, rates) -> None:
        price = pricing.get(model)
        assert price is not None
        assert (price.input, price.output, price.cache_read) == rates


class TestToolPrices:
    """A paid tool's price lives in the table, by the tool's name, per unit it bills (D56)."""

    def test_the_table_prices_the_two_search_tools(self) -> None:
        from ai_arch_toolkit.core import ToolPricing

        assert pricing.get_tool("brave_search") == ToolPricing(per_unit=0.005)
        assert pricing.get_tool("tavily_search") == ToolPricing(per_unit=0.008)
        assert pricing.get_tool("arxiv_search") is None
        assert {"brave_search", "tavily_search"} <= set(pricing.list_tools())

    def test_register_tool_sets_the_price_of_ones_own_plan(self) -> None:
        from ai_arch_toolkit.core import ToolPricing

        registry = PricingRegistry()
        registry.register_tool("brave_search", ToolPricing(per_unit=0.003))
        assert registry.get_tool("brave_search") == ToolPricing(per_unit=0.003)
        registry.unregister_tool("brave_search")
        assert registry.get_tool("brave_search") is None
        registry.reset()
        assert registry.get_tool("brave_search") == ToolPricing(per_unit=0.005)

    def test_a_tool_price_can_end_on_a_date(self) -> None:
        from ai_arch_toolkit.core import ToolPricing

        promo = ToolPricing(
            per_unit=0.004, until=date(2026, 11, 1), then=ToolPricing(per_unit=0.005)
        )
        registry = PricingRegistry()
        registry.register_tool("search", promo)

        assert registry.get_tool("search", on=date(2026, 11, 1)).per_unit == 0.004  # type: ignore[union-attr]
        assert registry.get_tool("search", on=date(2026, 11, 2)).per_unit == 0.005  # type: ignore[union-attr]

    def test_a_toml_file_takes_a_tools_section(self, tmp_path: Path) -> None:
        priced = tmp_path / "prices.toml"
        priced.write_text(
            "[tools.my_search]\nper_unit = 0.01\nuntil = 2026-12-31\n\n"
            "[tools.my_search.then]\nper_unit = 0.02\n"
        )
        registry = PricingRegistry()

        registry.load(priced)

        assert registry.get_tool("my_search", on=date(2026, 12, 31)).per_unit == 0.01  # type: ignore[union-attr]
        assert registry.get_tool("my_search", on=date(2027, 1, 1)).per_unit == 0.02  # type: ignore[union-attr]
        assert "tools" not in registry.list_models()

    def test_the_meter_prices_one_unit_of_a_priced_tool_and_nothing_of_another(self) -> None:
        from ai_arch_toolkit.core import Cost, Money, OperationRequest, Usage

        def request(name: str) -> OperationRequest:
            return OperationRequest(kind="tool", parent_span_id="run", metadata={"tool": name})

        assert pricing.price(request("brave_search"), Usage()) == Cost.known(Money.from_usd(0.005))
        assert pricing.price(request("arxiv_search"), Usage()) == Cost.known(Money.zero())
