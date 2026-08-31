# Four-channel illumination protocol

## Software contract

### Confirmed USB/firmware identity (2026-07-14)

The connected USB-only Octopus board was identified as `VID_0483:PID_5740` on
`COM11`. Its acknowledged `M115` reply reported `FIRMWARE_NAME:Marlin
bugfix-2.0.x`, `MACHINE_TYPE:Tablet Scanner`, and `PROTOCOL_VERSION:1.0`.
That historical generic identity is no longer accepted. No `M106` command was
sent during this identification check. A later USB-only check, with DTR/RTS
enabled, confirmed acknowledged explicit-off commands for `M106 P0 S0` through
`M106 P3 S0`. This proves parser/selector acceptance only; it does not prove
the physical header mapping until the revised firmware is compiled and flashed.

The release firmware now embeds the mapping-specific `M115` marker
`TS-LIGHT-V3-P0F0-P1F1-P2HE0-P3HE1-LOCK-RHOME`. Real-device discovery rejects firmware
without this exact marker so an older selector order cannot drive the lamps.

The software controls four logical channels: `uv255`, `uv310`, `uv365`, and
`vis`. Their selectors are locked to the mapping-specific firmware contract;
**Beállítások → Haladó** displays them for verification but cannot rotate them.

The intended electrical connections are:

| Channel | Octopus V1.1 header |
| --- | --- |
| 255 nm | HE0 |
| 310 nm | HE1 |
| 365 nm | FAN1 |
| VIS | FAN0 |

The fixed selector contract is `P0=FAN0/VIS`, `P1=FAN1/365 nm`,
`P2=HE0/255 nm`, and `P3=HE1/310 nm`. The firmware therefore uses
`FAN1_PIN` / `FAN2_PIN` / `FAN3_PIN` for FAN1 / HE0 / HE1 respectively in
`TabletScanner_Firmware_UV_MultiLED/Firmware/Marlin/src/pins/stm32f4/pins_BTT_OCTOPUS_V1_common.h`,
and `NUM_M106_FANS 4` in `Configuration.h`.

The firmware was compile-validated on 2026-07-22 with:

```text
cd TabletScanner_Firmware_UV_MultiLED/Firmware
python -m platformio run -e STM32F446ZE_btt
```

The generated, unflashed binary was rebuilt on 2026-07-28 with recoverable homing failures:
`.pio/build/STM32F446ZE_btt/firmware.bin` (SHA-256
`270597C4EC6E81492E2595B69202F95C2F1D1CCA8427A0E968614F0415119504`).
The successful build does not prove
physical output routing; verify it with a multimeter before connecting lamps.

The custom firmware maps those headers to the fixed `M106 P...` outputs. This
mapping must be verified with a current-limited dummy load before
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

Only one logical channel may be active. The backend sends P0–P3 OFF before one
ON, and the `-LOCK` firmware independently clears the other three fan speeds
whenever it receives a nonzero `M106`. Every timeout, failed capture, abort,
disconnect, and shutdown attempts an all-off command.

## Bring-up checklist

1. Confirm the Octopus firmware reports its approved identity through `M115`.
2. Confirm all four configured outputs are OFF at firmware boot.
3. Verify each fixed `M106` selector with a dummy load.
4. Configure UV brightness and timeout values before enabling UV lamps.
5. Test dimmed and full UV timeout behavior one channel at a time.
6. Verify all-off on application exit and controller disconnect.
7. With the selected axis motor disconnected before power-on, issue its `G28`, verify
   `Error:Homing failed`, then verify `M105` is acknowledged without resetting the board.
8. Guard the emergency stop and home Y once. Verify the CoreXY carriage travels toward the same
   physical sensor as before, the application reports `Y:2` after backoff, and a small positive-Y
   jog moves away from that sensor.

## Filter revolver protocol

Marlin exposes its internal fourth `I` axis as the operator-facing `A` axis. The configured
Hall-sensor endstop homes with `G28 A`; the six positions are 60° apart in the configured
0–360° range. A-axis homing uses 90°/s (5400°/min), with a 60-second backend timeout and a
70-second frontend request timeout. Marlin's explicit `Homing Failed` response terminates the
request immediately without treating the controller as disconnected. The `-RHOME` firmware
retains endstop validation but replaces Marlin's fatal `kill()` path for a missed endstop:
the failed axis remains unhomed, `G28` returns to the command loop, and other serial commands
remain available. Retrying motion on that axis still requires a successful homing operation.
Marlin retains the proven CoreXY motor mixing and max-homed native Y configuration. The backend
maps operator coordinates with `logical Y = 165 - native Y` for position reports, absolute
moves, and relative moves. Thus native `Y:173` after the 2 mm homing backoff is shown as
operator `Y:2`, and positive operator Y moves away from the home sensor without changing
CoreXY motor directions or the Zâ†’Yâ†’X homing sequence.
The Motor 6 DIAG jumper
must be removed because the external Hall sensor uses `DIAG6/E2DET` and sensorless homing is
disabled. After the fine Hall-sensor trigger, the circular-axis firmware continues 90° in the
homing direction—30° past the trigger plus the observed 60° slot-6-to-slot-1 step—to establish the slot-one
reference. The A/I min/max software endstops are disabled so the circular
revolver has no 0/360° travel seam; X/Y/Z software limits remain enabled. Manual control uses
acknowledged `G91`, `G1 A+/-60 F5400`, `M400`, and restores `G90`, giving normal rotation the
same 90°/s speed as the fast homing seek.
The screen-right (`up`) command sends `A-60` and advances `1→2`; the screen-left (`down`)
command sends `A+60` and moves `1→6`. At slot one, `G92 A0` normalizes the circular coordinate.
The backend changes the active one-based slot only after `M400` and mode restoration are
acknowledged. Disconnecting or changing the motion adapter invalidates the homed slot.

The filter position stored in a capture plan remains metadata only; automatic measurement does
not yet call the manual revolver controller.
