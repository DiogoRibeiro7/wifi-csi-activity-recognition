# Hardware Setup

This guide walks through configuring the currently supported WiFi chipsets.
Each section assumes the software environment from the installation guide is
ready.

## Intel 5300

1. Install the modified iwlwifi driver and firmware from the
   [Linux 5300 CSI Tool](https://github.com/dhalperi/linux-80211n-csitool).
2. Load the module and verify the interface with `iw dev`.
3. Attach external antennas if required and position them according to the
   experiment design.
4. Run the calibration utility:
   `python -m wifi_activity_recognition.cli calibrate --device intel5300`.

## ESP32

1. Flash the ESP32 board with the CSI-enabled firmware from the
   [Espressif CSI repo](https://github.com/espressif/esp32-wifi-csi).
2. Connect the board via USB and note the serial port.
3. Ensure the device is in monitor mode and streaming CSI packets.
4. Start the listener:
   `python -m wifi_activity_recognition.cli listen --device esp32 --port /dev/ttyUSB0`.

## Atheros AR9300

1. Install the `ath9k` driver and ensure the interface is in monitor mode.
2. Use the provided `hardware.atheros.AtherosReader` class to capture CSI:
   `python -m wifi_activity_recognition.cli listen --device atheros`.
3. If phase or amplitude offsets appear, run the calibration command with the
   `--device atheros` option.

After completing the steps above, you can capture CSI data and proceed to the
preprocessing or training pipelines.
