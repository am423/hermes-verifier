"""The report must state its own limits, every run, without being asked."""

from __future__ import annotations

from autoresearch.report import render


def test_the_caveat_is_always_present():
    text = render({"verdict": "no candidate cleared the bar",
                   "power": {"turns": 10, "sessions": 2, "positive_labels": 4}})
    assert "repair FOLLOWED" in text
    assert "not that a review would have prevented it" in text


def test_a_low_power_run_says_so():
    text = render({"verdict": "no candidate cleared the bar",
                   "power": {"turns": 40, "sessions": 3, "positive_labels": 5}})
    assert "UNDERPOWERED" in text, (
        "a null with few positive labels is not evidence, and the report must say so"
    )


def test_an_adequately_powered_run_does_not_cry_underpowered():
    text = render({"verdict": "no candidate cleared the bar",
                   "power": {"turns": 900, "sessions": 30, "positive_labels": 350}})
    assert "UNDERPOWERED" not in text
    assert "no candidate cleared the bar" in text


def test_the_verdict_is_stated_even_when_null():
    text = render({"verdict": "no candidate cleared the bar", "power": {}})
    assert "no candidate cleared the bar" in text


def test_the_underpowered_banner_follows_the_deciding_half():
    """A survivor is confirmed (or not) on the VALIDATION half, so that half
    decides; a null was decided by the FIT half. The banner must follow the
    half that decided, not the overall total."""
    strong = {"turns": 900, "sessions": 30, "positive_labels": 350}
    thin = {"turns": 120, "sessions": 9, "positive_labels": 5}
    # survivors exist -> validation decides; it is thin -> banner
    text = render({"verdict": "adopted: rm", "power": strong,
                   "fit_power": strong, "val_power": thin,
                   "survivors": ["rm"], "deciding_half": "validation",
                   "held_out": {"rm": {"confirmed": True}}})
    assert "UNDERPOWERED" in text
    assert "the validation half" in text
    # nothing survived -> the fit half decided; it is strong, so even though
    # the (unreached) validation half is thin, this is not underpowered
    text = render({"verdict": "no candidate cleared the bar", "power": strong,
                   "fit_power": strong, "val_power": thin,
                   "survivors": [], "deciding_half": "fit"})
    assert "UNDERPOWERED" not in text


def test_the_report_shows_the_bar_and_every_candidates_cluster_interval():
    strong = {"turns": 900, "sessions": 30, "positive_labels": 350}
    text = render({
        "verdict": "no candidate cleared the bar", "power": strong,
        "fit_power": strong, "val_power": strong, "survivors": [],
        "deciding_half": "fit",
        "multiplicity": {"tests": 12, "alpha_each": 0.0042, "bootstraps": 4000,
                         "quantile_draws": 8,
                         "warning": "only 8.3 draws below the lower quantile"},
        "fit_results": {
            "rm": {"gap": 0.4, "ci": (0.1, 0.7), "cluster_ci": (-0.2, 0.9),
                   "clears_bar": False},
            "rare": {"gap": None, "ci": (None, None), "cluster_ci": (None, None),
                     "clears_bar": False, "ineligible": "fires in 2 session(s)"},
        },
    })
    assert "alpha each" in text
    assert "0.0042" in text
    assert "cluster_ci" in text and "(-0.2, 0.9)" in text, (
        "each candidate's cluster interval must be rendered"
    )
    assert "rare" in text and "fires in 2 session(s)" in text
    assert "WARNING" in text
