"""Keep the advertised hardware list in step with the driver registry.

The public ``SUPPORTED_HARDWARE`` constant is maintained by hand while the
registry is populated at import time, so the two can drift apart silently.
That is exactly what happened: the constant advertised Broadcom and MediaTek
long after their registration blocks had been commented out.
"""

import pytest

import wifi_activity_recognition as pkg
from wifi_activity_recognition.hardware import list_supported_hardware


@pytest.mark.unit
def test_every_advertised_platform_is_actually_registered() -> None:
    """Nothing in SUPPORTED_HARDWARE may lack a registered driver."""
    registered = set(list_supported_hardware())
    advertised = set(pkg.SUPPORTED_HARDWARE)

    unbacked = advertised - registered
    assert not unbacked, (
        f"SUPPORTED_HARDWARE advertises {sorted(unbacked)}, which no registered "
        "driver provides. Either register a driver or move the entry to "
        "PLANNED_HARDWARE."
    )


@pytest.mark.unit
def test_planned_platforms_are_not_registered() -> None:
    """PLANNED_HARDWARE is for future work, so it must stay unregistered."""
    registered = set(list_supported_hardware())
    planned = set(pkg.PLANNED_HARDWARE)

    landed = planned & registered
    assert not landed, (
        f"{sorted(landed)} now has a registered driver but is still listed as "
        "planned. Promote it to SUPPORTED_HARDWARE."
    )


@pytest.mark.unit
def test_supported_and_planned_do_not_overlap() -> None:
    """A platform cannot be both shipped and planned."""
    overlap = set(pkg.SUPPORTED_HARDWARE) & set(pkg.PLANNED_HARDWARE)
    assert not overlap, f"listed as both supported and planned: {sorted(overlap)}"


@pytest.mark.unit
def test_package_info_reports_both_lists_separately() -> None:
    """get_package_info must not blur shipped and planned support."""
    info = pkg.get_package_info()

    assert info["supported_hardware"] == pkg.SUPPORTED_HARDWARE
    assert info["planned_hardware"] == pkg.PLANNED_HARDWARE

    for platform in pkg.PLANNED_HARDWARE:
        assert platform not in info["supported_hardware"]
