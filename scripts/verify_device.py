#!/usr/bin/env python3
"""Verify a real CSI capture device against the driver contract.

Run this with hardware attached. It is deliberately not part of the test suite:
CI has no devices, and a test that silently passes when the hardware is absent
would be worse than no test.

    python scripts/verify_device.py --hardware esp32 --port COM5
    python scripts/verify_device.py --hardware intel_5300
    python scripts/verify_device.py --hardware qualcomm --device-ip 192.168.1.42

Each check prints PASS, FAIL or SKIP with the observed value, and the script
exits non-zero if any check fails. Record the output against the matrix in
docs/hardware_verification.md.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Any, Callable

from wifi_activity_recognition.hardware import CSIReader, list_supported_hardware
from wifi_activity_recognition.hardware.base import CSIData, validate_csi_data

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_results: list[tuple[str, str, str]] = []


def check(name: str, fn: Callable[[], tuple[str, str]]) -> None:
    """Run one check and record its verdict and observation."""
    try:
        verdict, detail = fn()
    except Exception as exc:  # noqa: BLE001 - a raising check is a failing check
        verdict, detail = FAIL, f"{type(exc).__name__}: {exc}"
    _results.append((name, verdict, detail))
    colour = {PASS: "", FAIL: "", SKIP: ""}[verdict]
    print(f"  [{verdict}] {name}{colour}")
    if detail:
        print(f"         {detail}")


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble the driver config from CLI arguments."""
    params: dict[str, Any] = {"mode": "real"}
    if args.port:
        params["serial_port"] = args.port
    if args.baud:
        params["baud_rate"] = args.baud
    if args.device_ip:
        params["device_ip"] = args.device_ip
    return {
        "sampling_rate": args.sampling_rate,
        "channel": args.channel,
        "bandwidth": args.bandwidth,
        "additional_params": params,
    }


def main() -> int:
    """Run the verification sequence and summarise the outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", required=True, help="Registered driver name")
    parser.add_argument("--port", help="Serial port, e.g. COM5 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, help="Serial baud rate")
    parser.add_argument("--device-ip", help="Device address for network drivers")
    parser.add_argument("--packets", type=int, default=100, help="Packets to capture")
    parser.add_argument("--sampling-rate", type=float, default=100.0)
    parser.add_argument("--channel", type=int, default=6)
    parser.add_argument("--bandwidth", type=float, default=20.0)
    args = parser.parse_args()

    available = list_supported_hardware()
    if args.hardware not in available:
        print(f"Unknown hardware '{args.hardware}'. Registered: {available}")
        return 2

    print(f"\nVerifying '{args.hardware}' against a real device\n")
    reader = CSIReader(args.hardware, build_config(args))

    # 1. Connection ------------------------------------------------------
    def _connect() -> tuple[str, str]:
        ok = reader.connect()
        return (PASS if ok else FAIL, f"connect() -> {ok}")

    check("connects to the device", _connect)
    if not reader.is_connected:
        print("\nCannot continue without a connection.")
        return summarise()

    # 2. Hardware identity ----------------------------------------------
    check(
        "reports hardware info",
        lambda: (PASS, str(reader.get_hardware_info())[:200]),
    )

    # 3. Calibration -----------------------------------------------------
    def _calibrate() -> tuple[str, str]:
        ok = reader.calibrate()
        return (PASS if ok else SKIP, f"calibrate() -> {ok}")

    check("calibrates", _calibrate)

    # 4. Capture ---------------------------------------------------------
    reader.start_streaming()
    packets: list[CSIData] = []
    started = time.perf_counter()
    while len(packets) < args.packets and time.perf_counter() - started < 60:
        packet = reader.read_packet()
        if packet is not None:
            packets.append(packet)
    elapsed = time.perf_counter() - started
    reader.stop_streaming()

    check(
        f"captures {args.packets} packets",
        lambda: (
            PASS if len(packets) >= args.packets else FAIL,
            f"{len(packets)} packets in {elapsed:.1f}s",
        ),
    )
    if not packets:
        reader.disconnect()
        return summarise()

    # 5. Packet shape ----------------------------------------------------
    def _shape() -> tuple[str, str]:
        shapes = {p.shape for p in packets}
        return (
            PASS if len(shapes) == 1 else FAIL,
            f"shapes observed: {shapes}",
        )

    check("packet shape is stable", _shape)

    # 6. Physical plausibility -------------------------------------------
    def _valid() -> tuple[str, str]:
        bad = [i for i, p in enumerate(packets) if not validate_csi_data(p)]
        return (
            PASS if not bad else FAIL,
            f"{len(bad)} of {len(packets)} failed validation"
            + (f", first at index {bad[0]}" if bad else ""),
        )

    check("packets pass validation", _valid)

    # 7. Timestamps ------------------------------------------------------
    def _timing() -> tuple[str, str]:
        stamps = [p.timestamp for p in packets]
        if len(stamps) < 2:
            return SKIP, "not enough packets"
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        monotonic = all(g >= 0 for g in gaps)
        rate = 1.0 / statistics.fmean(gaps) if statistics.fmean(gaps) > 0 else 0.0
        return (
            PASS if monotonic else FAIL,
            f"monotonic={monotonic}, observed ~{rate:.1f} packets/s "
            f"(configured {args.sampling_rate})",
        )

    check("timestamps advance monotonically", _timing)

    # 8. Data varies -----------------------------------------------------
    def _varies() -> tuple[str, str]:
        first = packets[0].amplitude
        identical = sum(1 for p in packets[1:] if (p.amplitude == first).all())
        return (
            PASS if identical == 0 else FAIL,
            f"{identical} packets identical to the first "
            "(a frozen capture usually means a stalled reader)",
        )

    check("amplitude varies between packets", _varies)

    # 9. Clean shutdown --------------------------------------------------
    def _disconnect() -> tuple[str, str]:
        reader.disconnect()
        return (PASS if not reader.is_connected else FAIL, "disconnect() completed")

    check("disconnects cleanly", _disconnect)

    return summarise()


def summarise() -> int:
    """Print the tally and return an exit code."""
    failed = [name for name, verdict, _ in _results if verdict == FAIL]
    passed = sum(1 for _, verdict, _ in _results if verdict == PASS)
    skipped = sum(1 for _, verdict, _ in _results if verdict == SKIP)

    print(f"\n{passed} passed, {len(failed)} failed, {skipped} skipped")
    if failed:
        print("Failed checks: " + ", ".join(failed))
        return 1
    print("\nRecord this run in docs/hardware_verification.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
