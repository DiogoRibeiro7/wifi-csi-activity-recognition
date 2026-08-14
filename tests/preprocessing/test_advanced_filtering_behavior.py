"""Behaviour tests for the advanced filters.

Coverage on this module was 61%, but the shortfall was not evenly spread:
``median_filter``'s filtering body and the whole of ``morphological_filter``
never executed. The only median-filter test asserted that an even kernel size
raises, so the function was covered exclusively by its guard clause.

``multirate_resample`` had a passing test that used ``up=2, down=1`` on 8
subcarriers -- a ratio where floor and ceiling agree. Every other ratio raised,
because the declared subcarrier count was computed as ``n * up // down`` while
``resample_poly`` returns ``ceil(n * up / down)`` samples.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from wifi_activity_recognition.hardware.base import CSIData
from wifi_activity_recognition.preprocessing import (
    adaptive_wiener_filter,
    median_filter,
    morphological_filter,
    multirate_resample,
)


def _csi(amplitude: np.ndarray) -> CSIData:
    n_rx, n_tx, n_sc = amplitude.shape
    return CSIData(
        timestamp=0.0,
        amplitude=amplitude,
        phase=np.zeros_like(amplitude),
        frequency=5.0,
        bandwidth=20.0,
        n_tx=n_tx,
        n_rx=n_rx,
        n_subcarriers=n_sc,
    )


# ---------------------------------------------------------------------------
# median_filter -- previously only its guard clause ran
# ---------------------------------------------------------------------------


def test_median_filter_removes_an_impulse() -> None:
    """The purpose of a median filter: kill impulses, keep the baseline."""
    amplitude = np.ones((1, 1, 15))
    amplitude[0, 0, 7] = 100.0

    filtered = median_filter(_csi(amplitude), kernel_size=3).amplitude

    assert filtered[0, 0, 7] == pytest.approx(1.0), "impulse survived the filter"
    assert filtered[0, 0, :6] == pytest.approx(1.0)


def test_median_filter_preserves_a_step_edge() -> None:
    """Median filtering keeps edges that a mean filter would smear."""
    amplitude = np.concatenate([np.zeros(8), np.ones(8)]).reshape(1, 1, 16)

    filtered = median_filter(_csi(amplitude), kernel_size=3).amplitude[0, 0]

    assert set(np.unique(filtered)) <= {0.0, 1.0}, "edge was blurred to intermediates"


def test_median_filter_preserves_shape_and_leaves_input_alone() -> None:
    """Filtering is pure and shape-preserving."""
    amplitude = np.random.default_rng(0).normal(size=(2, 1, 9))
    original = amplitude.copy()
    csi = _csi(amplitude)

    filtered = median_filter(csi, kernel_size=3)

    assert filtered.amplitude.shape == amplitude.shape
    assert np.array_equal(csi.amplitude, original)


@pytest.mark.parametrize("bad_kernel", [0, -1, 2, 4])
def test_median_filter_rejects_invalid_kernel_sizes(bad_kernel: int) -> None:
    """Kernel size must be a positive odd integer."""
    with pytest.raises(ValueError, match="positive odd integer"):
        median_filter(_csi(np.ones((1, 1, 8))), kernel_size=bad_kernel)


# ---------------------------------------------------------------------------
# morphological_filter -- previously never executed at all
# ---------------------------------------------------------------------------


def test_morphological_opening_suppresses_a_narrow_peak() -> None:
    """Opening removes structures narrower than the window."""
    amplitude = np.zeros((1, 1, 20))
    amplitude[0, 0, 10] = 5.0

    opened = morphological_filter(_csi(amplitude), size=3, operation="opening")

    assert opened.amplitude[0, 0, 10] < 5.0, "narrow peak survived an opening"


def test_morphological_closing_fills_a_narrow_trough() -> None:
    """Closing is the dual: it removes narrow dips."""
    amplitude = np.full((1, 1, 20), 5.0)
    amplitude[0, 0, 10] = 0.0

    closed = morphological_filter(_csi(amplitude), size=3, operation="closing")

    assert closed.amplitude[0, 0, 10] > 0.0, "narrow trough survived a closing"


def test_opening_never_exceeds_and_closing_never_falls_below_the_input() -> None:
    """The defining extensivity property of these two operations."""
    amplitude = np.random.default_rng(0).normal(size=(1, 1, 32))
    csi = _csi(amplitude)

    opened = morphological_filter(csi, size=3, operation="opening").amplitude
    closed = morphological_filter(csi, size=3, operation="closing").amplitude

    assert np.all(opened <= amplitude + 1e-9), "opening is not anti-extensive"
    assert np.all(closed >= amplitude - 1e-9), "closing is not extensive"


@pytest.mark.parametrize("bad_size", [0, -3])
def test_morphological_filter_rejects_invalid_size(bad_size: int) -> None:
    """Window size must be positive."""
    with pytest.raises(ValueError, match="size must be positive"):
        morphological_filter(_csi(np.ones((1, 1, 8))), size=bad_size)


def test_morphological_filter_rejects_unknown_operation() -> None:
    """Only opening and closing are supported."""
    with pytest.raises(ValueError, match="opening.*closing"):
        morphological_filter(_csi(np.ones((1, 1, 8))), operation="erosion")


# ---------------------------------------------------------------------------
# multirate_resample -- the declared length disagreed with the real one
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.parametrize(
    "n_subcarriers,up,down",
    [(31, 1, 2), (30, 1, 4), (33, 1, 2), (30, 3, 4), (30, 2, 4), (8, 2, 1)],
)
def test_resample_declares_the_length_it_actually_produced(
    n_subcarriers: int, up: int, down: int
) -> None:
    """n_subcarriers must match the resampled array for every ratio.

    Previously computed as ``n * up // down`` while ``resample_poly`` returns
    ``ceil(n * up / down)``, so any inexact ratio raised from CSIData's own
    shape validation.
    """
    amplitude = np.random.default_rng(0).normal(size=(1, 1, n_subcarriers))
    expected = len(signal.resample_poly(np.zeros(n_subcarriers), up, down))

    result = multirate_resample(_csi(amplitude), up=up, down=down)

    assert result.amplitude.shape[-1] == expected
    assert result.n_subcarriers == expected
    assert result.phase.shape == result.amplitude.shape


def test_resample_accepts_a_non_sequence_iterable_of_fields() -> None:
    """``fields`` is typed Iterable, so a generator must work.

    The old implementation indexed ``fields[0]``, which raises for a generator.
    """
    csi = _csi(np.random.default_rng(0).normal(size=(1, 1, 16)))
    result = multirate_resample(
        csi, up=1, down=2, fields=(name for name in ("amplitude", "phase"))
    )
    assert result.n_subcarriers == result.amplitude.shape[-1]


def test_resample_rejects_field_sets_that_would_desync_amplitude_and_phase() -> None:
    """Resampling only one field leaves an unconstructable CSIData."""
    csi = _csi(np.ones((1, 1, 16)))
    with pytest.raises(ValueError, match="both"):
        multirate_resample(csi, up=1, down=2, fields=("amplitude",))


def test_resample_rejects_empty_fields() -> None:
    """An empty field set has nothing to resample."""
    with pytest.raises(ValueError, match="at least one field"):
        multirate_resample(_csi(np.ones((1, 1, 8))), up=1, down=2, fields=())


@pytest.mark.parametrize("up,down", [(0, 1), (1, 0), (-1, 2)])
def test_resample_rejects_non_positive_rates(up: int, down: int) -> None:
    """Rational resampling needs positive integers on both sides."""
    with pytest.raises(ValueError, match="positive integers"):
        multirate_resample(_csi(np.ones((1, 1, 8))), up=up, down=down)


# ---------------------------------------------------------------------------
# adaptive_wiener_filter argument validation -- guards were never exercised
# ---------------------------------------------------------------------------


def test_wiener_rejects_non_positive_window() -> None:
    """Window size must be positive."""
    with pytest.raises(ValueError, match="mysize must be positive"):
        adaptive_wiener_filter(_csi(np.ones((1, 1, 8))), mysize=0)


def test_wiener_rejects_negative_noise_power() -> None:
    """Noise power is a variance and cannot be negative."""
    with pytest.raises(ValueError, match="noise must be non-negative"):
        adaptive_wiener_filter(_csi(np.ones((1, 1, 8))), noise=-1.0)


@pytest.mark.parametrize("bad_axis", [3, -4])
def test_wiener_rejects_out_of_range_axis(bad_axis: int) -> None:
    """An axis outside the array's rank must be reported, not indexed."""
    with pytest.raises(ValueError, match="axis out of range"):
        adaptive_wiener_filter(_csi(np.ones((1, 1, 8))), axis=bad_axis)
