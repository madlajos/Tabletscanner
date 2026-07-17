# Four-channel illumination protocol

## Software contract

### Confirmed USB/firmware identity (2026-07-14)

The connected USB-only Octopus board was identified as `VID_0483:PID_5740` on
`COM11`. Its acknowledged `M115` reply reports `FIRMWARE_NAME:Marlin
bugfix-2.0.x`, `MACHINE_TYPE:Tablet Scanner`, and `PROTOCOL_VERSION:1.0`.
The existing discovery rule is therefore compatible. No `M106` command was
sent during this identification check. A later USB-only check, with DTR/RTS
enabled, confirmed acknowledged explicit-off commands for `M106 P0 S0` through
`M106 P3 S0`. This proves parser/selector acceptance only; it does not prove
the physical header mapping until the revised firmware is compiled and flashed.

The software controls four logical channels: `uv255`, `uv310`, `uv365`, and
`vis`. The actual `M106` output selectors are deliberately not hard-coded.
An operator configures them in **Beállítások → Haladó** as values such as `P0`.

The intended electrical connections are:

| Channel | Octopus V1.1 header |
| --- | --- |
| 255 nm | HE0 |
| 310 nm | HE1 |
| 365 nm | HE2 |
| VIS | FAN0 |

The configured Marlin selector contract is `P0=FAN0/VIS`, `P1=HE0/255 nm`,
`P2=HE1/310 nm`, and `P3=HE2/365 nm`. This requires the `FAN1_PIN`,
`FAN2_PIN`, and `FAN3_PIN` reassignment in
`TabletScanner_Firmware_UV_MultiLED/Firmware/Marlin/src/pins/stm32f4/pins_BTT_OCTOPUS_V1_common.h`,
and `NUM_M106_FANS 4` in `Configuration.h`.

The firmware was compile-validated on 2026-07-14 with:

```text
cd TabletScanner_Firmware_UV_MultiLED/Firmware
python -m platformio run -e STM32F446ZE_btt
```

The generated, unflashed binary is
`.pio/build/STM32F446ZE_btt/firmware.bin`. The successful build does not prove
physical output routing; verify it with a multimeter before connecting lamps.

The custom firmware must map those headers to the configured `M106 P...`
outputs. This mapping must be verified with a current-limited dummy load before
any UV lamp is connected.

## Commands and safety

The controller sends acknowledged commands in this form:

```text
M106 P<number> S<pwm>
```

`S0` turns a configured channel off. For UV channels, PWM is calculated only on
the backend as `round(255 * percentage / 100)`; therefore 10% is `S26` and
100% is `S255`. UV single-click uses dimmed output; double-click uses full
output. Each mode has its own configured automatic shutoff time. VIS is always
100%, uses single-click control, and has no thermal timer.

Only one logical channel may be active. Every timeout, failed capture, abort,
disconnect, and shutdown attempts an all-off command.

## Bring-up checklist

1. Confirm the Octopus firmware reports its approved identity through `M115`.
2. Confirm all four configured outputs are OFF at firmware boot.
3. Configure selectors and verify each `M106` command with a dummy load.
4. Configure UV brightness and timeout values before enabling UV lamps.
5. Test dimmed and full UV timeout behavior one channel at a time.
6. Verify all-off on application exit and controller disconnect.

## Filter revolver protocol

Marlin exposes its internal fourth `I` axis as the operator-facing `A` axis. The configured
Hall-sensor endstop homes with `G28 A`; the six positions are 60° apart in the configured
0–360° range. Manual control uses acknowledged `G91`, `G1 A+/-60`, `M400`, and restores `G90`.
At the wrap boundary, `G92 A360` or `G92 A0` keeps the move inside the configured soft range.
The backend changes the active one-based slot only after `M400` and mode restoration are
acknowledged. Disconnecting or changing the motion adapter invalidates the homed slot.

The filter position stored in a capture plan remains metadata only; automatic measurement does
not yet call the manual revolver controller.
