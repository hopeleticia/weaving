# Weaving Project

AI-based anomaly and fault-pattern analysis for Wintex direct weaving / smart creel PLC data.

## Project Purpose

This project investigates whether PLC signal behavior can reveal recognizable patterns before or during weaving alarms. The long-term goal is to build an anomaly/fault detection and prediction model.

Main objectives:

- preprocess PLC logs using the correct INT16 conversion and feature scaling rules
- inspect time-series behavior around alarm codes
- analyze signal relationships before, during, and after alarms
- compare baseline anomaly/fault detection models before data augmentation
- identify whether clear alarm-related signal patterns exist before creating augmented fault data

## Alarm Codes

Known alarm-code meanings used in the analysis:

| Alarm Code | Meaning |
|---:|---|
| 0 | Normal / no alarm |
| 2 | Weft yarn break detected |
| 4 | End mark |
| 8 | Warp yarn break detected |
| 10 | Weft yarn break + warp yarn break |


## Important Signals

Signals repeatedly used in the analysis include:

- `Motor_RPM`
- `Weaving_Line_Speed`
- `Infeed_Speed_PV`
- `INFEED_TORQUE`
- `Dancer_Pressure_PV`
- `Dancer_Gyro_Position`
- `Loadcell_PV`
- `Loadcell_Weight_PV`
- `Temperature_PV`
- `Humidity_PV`

## Data Processing

Raw PLC values must be converted before analysis:

1. Convert unsigned INT16-style values to signed:

   ```text
   if value > 32767, subtract 65536
   ```

2. Apply feature-specific scaling from the Wintex smart creel tag list:

   ```text
  260611 윈텍스 스마트크릴 태그리스트.pdf
   ```



