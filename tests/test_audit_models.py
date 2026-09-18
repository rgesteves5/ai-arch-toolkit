"""The owner-run model audit: the comparison is pure and runs offline."""

from __future__ import annotations

from scripts.audit_models import Row, audit, render, unlisted

from ai_arch_toolkit.core import ModelPricing, PricingRegistry


def registry() -> PricingRegistry:
    reg = PricingRegistry()
    reg.reset()
    for model in reg.list_models():
        reg.unregister(model)
    reg.register("acme-1", ModelPricing(input=1.0))
    reg.register("local-", ModelPricing(), match="prefix")
    return reg


def test_every_listed_id_says_how_it_is_priced() -> None:
    listed = {"acme": ["acme-1", "acme-1-20260101", "acme-1-mini", "local-llama"]}
    assert audit(listed, registry()) == [
        Row("acme", "acme-1", "exact"),
        Row("acme", "acme-1-20260101", "snapshot"),
        Row("acme", "acme-1-mini", "none"),
        Row("acme", "local-llama", "prefix"),
    ]


def test_table_ids_no_provider_lists_are_reported() -> None:
    reg = registry()
    listed = {"acme": ["acme-1-mini"]}
    assert unlisted(listed, reg) == ["acme-1", "local-"]
    report = render(audit(listed, reg), unlisted(listed, reg))
    assert "| acme | `acme-1-mini` | none |" in report
    assert "1 listed ids without a price." in report
