# Hardware Setup

This guide walks through configuring the currently supported WiFi chipsets.
Each section assumes the software environment from the installation guide is
ready and highlights platform‑specific notes and troubleshooting tips.

## Intel 5300

1. Install the modified iwlwifi driver and firmware from the
   [Linux 5300 CSI Tool](https://github.com/dhalperi/linux-80211n-csitool).
   On Ubuntu, the provided build scripts compile both driver and firmware.
2. Load the module and verify the interface with `iw dev`. If the interface
   does not appear, ensure Secure Boot is disabled and the module is signed.
3. Attach external antennas if required and position them according to the
   experiment design.
4. Run the calibration utility:
   `python -m wifi_activity_recognition.cli calibrate --device intel5300`.

**Troubleshooting**

- *"Operation not permitted"*: run driver commands with `sudo`.
- *Missing firmware*: copy firmware files to `/lib/firmware` and reload the
  module.

## ESP32

1. Flash the ESP32 board with the CSI-enabled firmware from the
   [Espressif CSI repo](https://github.com/espressif/esp32-wifi-csi).
2. Connect the board via USB and note the serial port (`COMx` on Windows,
   `/dev/ttyUSBx` on Linux, `/dev/tty.SLAB_USBtoUART` on macOS).
3. Ensure the device is in monitor mode and streaming CSI packets.
4. Start the listener:
   `python -m wifi_activity_recognition.cli listen --device esp32 --port /dev/ttyUSB0`.

**Troubleshooting**

- *Port not found*: check `dmesg` (Linux) or Device Manager (Windows) for
  correct driver installation.
- *No packets*: verify the firmware build and that the board is within range of
  the transmitter.

## Atheros AR9300

1. Install the `ath9k` driver and ensure the interface is in monitor mode.
2. Use the provided `hardware.atheros.AtherosReader` class to capture CSI:
   `python -m wifi_activity_recognition.cli listen --device atheros`.
3. If phase or amplitude offsets appear, run the calibration command with the
   `--device atheros` option.

**Troubleshooting**

- *Interface fails to enter monitor mode*: stop NetworkManager and reload the
  driver.
- *Inconsistent timestamps*: verify system clock and disable power saving
  features.

## Verifying Packet Capture

After configuring a device, confirm that CSI frames arrive:

```bash
python -m wifi_activity_recognition.cli listen --device <device> --duration 5
```

The command should report the number of packets captured. You can further
inspect raw packets with tools like `tcpdump` or `Wireshark`.

With hardware confirmed, proceed to the preprocessing or training pipelines.
