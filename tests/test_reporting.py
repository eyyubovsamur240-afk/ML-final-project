"""Tests for the LaTeX macro helpers in src.reporting."""

from src import config, reporting


def test_scope_dropped_categories_follows_config(monkeypatch):
    counts = {"yeni tikili": 10, "kohne tikili": 5, "torpaq": 3, "ofis": 1}
    monkeypatch.setattr(config, "KEEP_CATEGORIES", ["Yeni tikili", "Köhnə tikili"])
    assert reporting.scope_dropped_categories(counts) == "torpaq 3, ofis 1"
    monkeypatch.setattr(config, "KEEP_CATEGORIES", ["Yeni tikili", "Köhnə tikili", "Torpaq"])
    assert reporting.scope_dropped_categories(counts) == "ofis 1"
    assert reporting.scope_dropped_categories(None) == "none"
