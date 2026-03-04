"""Tests for vector index implementations."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.toolkit.memory.graph._index import BruteForceIndex, _cosine_similarity


class TestCosine:
    def test_identical_vectors(self):
        assert _cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 1]) == pytest.approx(0.0)


class TestBruteForceIndex:
    async def test_add_and_count(self):
        idx = BruteForceIndex()
        await idx.add("a", [1.0, 0.0])
        await idx.add("b", [0.0, 1.0])
        assert await idx.count() == 2

    async def test_search(self):
        idx = BruteForceIndex()
        await idx.add("a", [1.0, 0.0])
        await idx.add("b", [0.0, 1.0])
        await idx.add("c", [0.9, 0.1])
        results = await idx.search([1.0, 0.0], k=2)
        assert len(results) == 2
        # "a" should be most similar to [1, 0]
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0)

    async def test_update(self):
        idx = BruteForceIndex()
        await idx.add("a", [1.0, 0.0])
        await idx.update("a", [0.0, 1.0])
        results = await idx.search([0.0, 1.0], k=1)
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0)

    async def test_remove(self):
        idx = BruteForceIndex()
        await idx.add("a", [1.0, 0.0])
        await idx.remove("a")
        assert await idx.count() == 0

    async def test_remove_nonexistent(self):
        idx = BruteForceIndex()
        await idx.remove("nope")  # should not raise
        assert await idx.count() == 0
