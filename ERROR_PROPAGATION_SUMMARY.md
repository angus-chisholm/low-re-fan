# Error Propagation Summary

## Overview
This document details the error propagation calculations implemented in `serial_save_plot.py` using available sensor measurements and their standard deviations.

---

## 1. Density (ρ) - Lines 213-219

### Formula
$$\rho = \frac{P}{R \cdot T}$$

### Error Propagation
$$\sigma_\rho = \rho \cdot \frac{\sigma_T}{T}$$

**Note:** Pressure and gas constant R are fixed (not variable), so they don't contribute to uncertainty.

Where:
- $T$ = Absolute temperature (K) = 273.15 + temp_mean
- $\sigma_T$ = Temperature standard deviation (from sensor measurements)

### Implementation
```python
T_abs_mean = 273.15 + temp_mean
rho_std = rho_mean * (temp_std / T_abs_mean) if T_abs_mean != 0 else 0
```

---

## 2. Axial Velocity (Vₓ) - METHOD 1: Pitot-based (Lines 221-228)

### Formula
$$V_x = \sqrt{\frac{2 \Delta P}{\rho}}$$

### Error Propagation
$$\sigma_{V_x} = V_x \cdot \sqrt{\frac{1}{4}\left(\frac{\sigma_{\Delta P}}{\Delta P}\right)^2 + \frac{1}{4}\left(\frac{\sigma_\rho}{\rho}\right)^2}$$

Where:
- $\Delta P$ = Venturi pressure differential (sensor0)
- $\sigma_{\Delta P}$ = Sensor0 standard deviation

### Implementation
```python
rel_var_dp0 = (dp_sensor0_std / dp_sensor0_mean)**2 if dp_sensor0_mean != 0 else 0
rel_var_rho_local = (rho_std / rho_mean)**2 if rho_mean != 0 else 0
axvel_std = axvel_mean * np.sqrt(0.25 * rel_var_dp0 + 0.25 * rel_var_rho_local) if axvel_mean != 0 else 0
```

---

## 2b. Axial Velocity (Vₓ) - METHOD 2: Venturi static tappings only (Commented, Lines 233-237)

### Formula (Alternative)
$$\dot{m} = C \cdot E \cdot \epsilon \cdot \frac{\pi d^2}{4} \sqrt{2 \Delta P \cdot \rho}$$

$$V_x = \frac{\dot{m}}{A_{fan} \cdot \rho}$$

### Error Propagation
$$\sigma_{V_x} = V_x \cdot \sqrt{\left(\frac{\sigma_{\dot{m}}}{\dot{m}}\right)^2 + \left(\frac{\sigma_\rho}{\rho}\right)^2}$$

Where the mass flow error from the formula:
$$\sigma_{\dot{m}} = \dot{m} \cdot \sqrt{\frac{1}{4}\left(\frac{\sigma_{\Delta P}}{\Delta P}\right)^2 + \frac{1}{4}\left(\frac{\sigma_\rho}{\rho}\right)^2}$$

**Status:** Commented out - can be enabled by uncommenting lines 233-237

---

## 3. Mass Flow Rate (ṁ) - METHOD 1: Pitot-based (Lines 230-232)

### Formula
$$\dot{m} = \rho \cdot A_{throat} \cdot V_x$$

### Error Propagation
$$\sigma_{\dot{m}} = \dot{m} \cdot \sqrt{\left(\frac{\sigma_\rho}{\rho}\right)^2 + \left(\frac{\sigma_{V_x}}{V_x}\right)^2}$$

Where:
- $A_{throat}$ = Venturi throat area (fixed constant: $\pi d^2/4$)

### Implementation
```python
rel_var_axvel = (axvel_std / axvel_mean)**2 if axvel_mean != 0 else 0
mdot_std = mdot_mean * np.sqrt(rel_var_rho_local + rel_var_axvel) if mdot_mean != 0 else 0
```

---

## 3b. Mass Flow Rate (ṁ) - METHOD 2: Venturi static tappings (Commented, Lines 233-237)

### Formula (Alternative)
Same as shown in Section 2b - venturi formula with square root of $2 \Delta P \cdot \rho$

### Error Propagation
$$\sigma_{\dot{m}} = \dot{m} \cdot \sqrt{\frac{1}{4}\left(\frac{\sigma_{\Delta P}}{\Delta P}\right)^2 + \frac{1}{4}\left(\frac{\sigma_\rho}{\rho}\right)^2}$$

**Status:** Commented out - can be enabled by uncommenting lines 233-237

---

## Additional Derived Values (Already Implemented)

### Stage Pressure Rise (Lines 239-246)
$$\Delta P_{stage} = \Delta P_{sensor1} - 0.5 \rho V_x^2$$

$$\sigma_{\Delta P_{stage}} = \sqrt{\sigma_{\Delta P_{sensor1}}^2 + (0.5 V_x^2 \sigma_\rho)^2 + (\rho V_x \sigma_{V_x})^2}$$

### Flow Coefficient (Φ = Vₓ/U)
$$\sigma_\Phi = \Phi \cdot \sqrt{\left(\frac{\sigma_{V_x}}{V_x}\right)^2 + \left(\frac{\sigma_U}{U}\right)^2}$$

### Pressure Rise Coefficient (Ψ = ΔP_stage / (ρU²))
$$\sigma_\Psi = \Psi \cdot \sqrt{\left(\frac{\sigma_{\Delta P}}{\Delta P}\right)^2 + \left(\frac{\sigma_\rho}{\rho}\right)^2 + 4\left(\frac{\sigma_U}{U}\right)^2}$$

---

## Data Flow & CSV Output

All calculated means and standard deviations are saved to CSV with the following columns:
- `timestamp`, `point`
- `rpm_mean`, `rpm_stddev`
- `dp_venturi_mean`, `dp_venturi_stddev`
- `dp_stage_mean`, `dp_stage_stddev`
- `rho_mean`, `rho_stddev`
- `mflow_mean`, `mflow_stddev`
- `axvelocity_mean`, `axvelocity_stddev`
- `flow_coefficient_mean`, `flow_coefficient_stddev`
- `pressure_rise_coefficient_mean`, `pressure_rise_coefficient_stddev`

---

## Key Assumptions & Notes

1. **Constants:** Atmospheric pressure and gas constant are treated as fixed (no uncertainty)
2. **Relative Errors:** When a variable is zero or near-zero, relative error contributions are set to 0 to prevent division issues
3. **Pitot Method (Active):** Uses measured pressure differentials directly from dual sensors
4. **Venturi Method (Commented):** Alternative calculation using venturi calibration coefficient (C, E, ε)
5. **Temperature:** Primary source of density uncertainty; affects all calculated parameters
6. **RPM Uncertainty:** Propagates through blade tip velocity (U) to both flow coefficient (Φ) and pressure coefficient (Ψ)

