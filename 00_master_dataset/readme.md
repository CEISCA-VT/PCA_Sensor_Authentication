📊 1. Frequency (Hz)

What it is: The frequency of the AC signal being analyzed.

Unit: Hertz (Hz)

Purpose: Determines how circuit elements (especially reactive components like inductors and capacitors) behave at different signal rates.

🔊 2. Trace M (dB)

What it is: Magnitude of impedance (|Z|) expressed in decibels.

Formula: 20 * log10(|Z|)

Purpose: Shows how impedance varies logarithmically, often used in Bode plots or impedance sweeps.

🎯 3. Trace th (deg) (Theta)

What it is: Phase angle of the impedance.

Unit: Degrees (°)

Purpose: Shows the phase shift between voltage and current due to reactive elements.

⚡ 4. Trace |Z| (Ohm)

What it is: Magnitude of the total impedance.

Formula: |Z| = sqrt(R^2 + X^2)

Unit: Ohms (Ω)

Purpose: Represents how much the circuit resists the flow of alternating current.

🔥 5. Trace Rs (Ohm) (Series Resistance)

What it is: Real part of the series impedance.

Unit: Ohms (Ω)

Purpose: Represents energy dissipated (resistive losses) in a series model.

🌀 6. Trace Xs (Ohm) (Series Reactance)

What it is: Imaginary part of the series impedance.

Unit: Ohms (Ω)

Purpose: Represents inductive or capacitive reactance in a series configuration.

Positive: Inductive

Negative: Capacitive

🧮 7. Trace Rp (Ohm) (Parallel Resistance)

What it is: Real part of the impedance in a parallel model.

Unit: Ohms (Ω)

Purpose: Equivalent resistance when modeling the impedance as a parallel RLC circuit.

🔁 8. Trace Xp (Ohm) (Parallel Reactance)

What it is: Imaginary part of the parallel impedance.

Unit: Ohms (Ω)

Purpose: Reactance component in the parallel model.

Like Xs: Positive = Inductive, Negative = Capacitive

🔌 9. Trace Gp (S) (Parallel Conductance)

What it is: Real part of admittance (Y) in a parallel model.

Unit: Siemens (S)

Formula: Gp = 1 / Rp

Purpose: Conductance of the parallel resistor.

⚖️ 10. Trace Bp (S) (Parallel Susceptance)

What it is: Imaginary part of admittance (Y).

Unit: Siemens (S)

Formula: Bp = 1 / Xp

Purpose: Represents how easily the circuit allows AC due to reactive components.

🧲 11. Trace Ls (H) (Series Inductance)

What it is: Equivalent series inductance.

Unit: Henry (H)

Formula: Ls = Xs / (2πf) if Xs is inductive

Purpose: Inductive behavior modeled in series.

🌐 12. Trace Lp (H) (Parallel Inductance)

What it is: Equivalent parallel inductance.

Unit: Henry (H)

Formula: Lp = 1 / (2πf * Bp) if Bp is inductive

Purpose: Inductive behavior modeled in parallel.

🔋 13. Trace Vrms (V) (Voltage RMS)

What it is: Root-mean-square voltage across the device.

Unit: Volts (V)

Purpose: Represents effective AC voltage.

➕ 14. Trace Vreal (V)

What it is: Real part of the voltage phasor.

Unit: Volts (V)

Purpose: Component of voltage in-phase with reference (cosine component).

➖ 15. Trace Vimag (V)

What it is: Imaginary part of the voltage phasor.

Unit: Volts (V)

Purpose: Component of voltage 90° out of phase (sine component).

🔌 16. Trace Irms (A) (Current RMS)

What it is: Root-mean-square current.

Unit: Amperes (A)

Purpose: Effective current in the circuit.

🔄 17. Trace Ireal (A)

What it is: Real part of the current phasor.

Unit: Amperes (A)

Purpose: In-phase current with voltage.

🔃 18. Trace Iimag (A)

What it is: Imaginary part of the current phasor.

Unit: Amperes (A)

Purpose: Out-of-phase component (reactive current).

📘 Summary Table
Parameter	Description	Unit
Frequency	Signal frequency	Hz
Trace M	Magnitude of impedance in dB	dB
Trace th	Phase angle of impedance	°
Trace |Z|	Impedance magnitude	Ohm (Ω)
Trace Rs	Series resistance	Ohm (Ω)
Trace Xs	Series reactance	Ohm (Ω)
Trace Rp	Parallel resistance	Ohm (Ω)
Trace Xp	Parallel reactance	Ohm (Ω)
Trace Gp	Parallel conductance	Siemens
Trace Bp	Parallel susceptance	Siemens
Trace Ls	Series inductance	Henry (H)
Trace Lp	Parallel inductance	Henry (H)
Trace Vrms	RMS voltage	Volts (V)
Trace Vreal	Real part of voltage	Volts (V)
Trace Vimag	Imaginary part of voltage	Volts (V)
Trace Irms	RMS current	Amps (A)
Trace Ireal	Real part of current	Amps (A)
Trace Iimag	Imaginary part of current	Amps (A)


🧾 Explanation of Each Parameter
Parameter Name	Meaning / Description	Units	Details
Frequency (Hz)	Frequency of the signal	Hz	AC frequency at which the measurement is taken. Affects the behavior of reactive components (L, C).
Trace M (dB)	Magnitude in decibels	dB	`M = 20 × log10(
Trace th (deg)	Phase angle (theta)	°	Phase shift between voltage and current. 0° means purely resistive, >0° inductive, <0° capacitive.
**Trace	Z	(Ohm)**	Impedance magnitude
Trace Rs (Ohm)	Series resistance	Ω	Real part of series impedance; represents resistive losses.
Trace Xs (Ohm)	Series reactance	Ω	Imaginary part of series impedance: Xs > 0 → inductive, Xs < 0 → capacitive.
Trace Rp (Ohm)	Parallel resistance	Ω	Real part of equivalent impedance in a parallel model.
Trace Xp (Ohm)	Parallel reactance	Ω	Imaginary part of impedance in a parallel model.
Trace Gp (S)	Parallel conductance	S	Gp = 1 / Rp, real part of admittance (Y = 1/Z).
Trace Bp (S)	Parallel susceptance	S	Bp = 1 / Xp, imaginary part of admittance. Positive = capacitive, negative = inductive.
Trace Ls (H)	Series inductance	H	Ls = Xs / (2πf), only if Xs > 0.
Trace Lp (H)	Parallel inductance	H	Lp = 1 / (2πf × Bp) if Bp < 0.
Trace Cs (F)	Series capacitance	F	Cs = -1 / (2πf × Xs) if Xs < 0. Models capacitor in series circuit.
Trace Cp (F)	Parallel capacitance	F	Cp = Bp / (2πf) if Bp > 0. Models capacitor in parallel circuit.
Trace Vrms (V)	RMS voltage	V	Root Mean Square value of voltage across DUT (Device Under Test).
Trace Vreal (V)	Real part of voltage	V	In-phase component (with respect to reference signal).
Trace Vimag (V)	Imaginary part of voltage	V	90° out-of-phase component (reactive part).
Trace Irms (A)	RMS current	A	RMS current flowing through the DUT.
Trace Ireal (A)	Real part of current	A	In-phase component of the current.
Trace Iimag (A)	Imaginary part of current	A	Out-of-phase component of the current.
📘 Key Concepts

Impedance (Z): Combines resistance (R) and reactance (X):

𝑍
=
𝑅
+
𝑗
𝑋
Z=R+jX

Admittance (Y): Inverse of impedance:

𝑌
=
𝐺
+
𝑗
𝐵
=
1
𝑍
Y=G+jB=
Z
1
	​


Series vs Parallel Models: Some measurements or systems are modeled with components in series (R + jX), while others use a parallel model (R || jX).

Inductive or Capacitive?

Xs > 0 or Bp < 0 → Inductive

Xs < 0 or Bp > 0 → Capacitive

Conversions:

Inductance from Reactance:

𝐿
=
𝑋
2
𝜋
𝑓
L=
2πf
X
	​


Capacitance from Reactance:

𝐶
=
−
1
2
𝜋
𝑓
𝑋
C=
2πfX
−1
	​

📉 Example Applications

Impedance analyzers / LCR meters often output all these parameters across a sweep of frequencies.

Useful in:

Component characterization (e.g., testing an inductor or capacitor)

PCB or antenna modeling

Battery or sensor impedance analysis

Biomedical impedance (e.g., tissue analysis)



# 📘 Impedance Measurement Parameters Explained

This document provides a detailed explanation of the electrical parameters typically output from an impedance analyzer, LCR meter, or simulation. These include resistive, reactive, and phasor-related quantities in both series and parallel configurations.

---

## 📊 Measurement Parameters

| **Parameter**         | **Description**                                     | **Units**   |
|------------------------|-----------------------------------------------------|-------------|
| `Frequency (Hz)`       | Signal frequency                                    | Hz          |
| `Trace M (dB)`         | Magnitude of impedance in decibels                 | dB          |
| `Trace th (deg)`       | Phase angle of impedance (theta)                   | °           |
| `Trace |Z| (Ohm)`      | Impedance magnitude                                | Ohms (Ω)    |
| `Trace Rs (Ohm)`       | Series resistance (real part of series Z)          | Ohms (Ω)    |
| `Trace Xs (Ohm)`       | Series reactance (imaginary part of series Z)      | Ohms (Ω)    |
| `Trace Rp (Ohm)`       | Parallel resistance                                | Ohms (Ω)    |
| `Trace Xp (Ohm)`       | Parallel reactance                                 | Ohms (Ω)    |
| `Trace Gp (S)`         | Parallel conductance (`Gp = 1 / Rp`)               | Siemens (S) |
| `Trace Bp (S)`         | Parallel susceptance (`Bp = 1 / Xp`)               | Siemens (S) |
| `Trace Ls (H)`         | Equivalent series inductance                       | Henry (H)   |
| `Trace Lp (H)`         | Equivalent parallel inductance                     | Henry (H)   |
| `Trace Cs (F)`         | Equivalent series capacitance                      | Farads (F)  |
| `Trace Cp (F)`         | Equivalent parallel capacitance                    | Farads (F)  |
| `Trace Vrms (V)`       | RMS voltage across DUT                             | Volts (V)   |
| `Trace Vreal (V)`      | Real (in-phase) component of voltage               | Volts (V)   |
| `Trace Vimag (V)`      | Imaginary (quadrature) component of voltage        | Volts (V)   |
| `Trace Irms (A)`       | RMS current through DUT                            | Amps (A)    |
| `Trace Ireal (A)`      | Real (in-phase) component of current               | Amps (A)    |
| `Trace Iimag (A)`      | Imaginary (quadrature) component of current        | Amps (A)    |

---

## 🧠 Key Concepts

### 🔌 Impedance and Admittance
- **Impedance (Z):**  
  \[
  Z = R + jX
  \]  
  Combines resistance (`R`) and reactance (`X`) to describe opposition to AC.

- **Admittance (Y):**  
  \[
  Y = G + jB = \frac{1}{Z}
  \]  
  The inverse of impedance, describing how easily AC flows.

### ⚡ Series vs Parallel Models
- **Series Model:**  
  Assumes components are arranged in a chain (R + jX).
- **Parallel Model:**  
  Models impedance as a resistor and reactance in parallel (R || jX).

---

## 📐 Inductive or Capacitive Behavior

| Reactance / Susceptance | Component Type |
|--------------------------|----------------|
| `Xs > 0` or `Bp < 0`     | Inductive      |
| `Xs < 0` or `Bp > 0`     | Capacitive     |

---

## 🔁 Conversion Formulas

- **Series Inductance:**  
  \[
  L_s = \frac{X_s}{2\pi f}
  \]

- **Series Capacitance:**  
  \[
  C_s = \frac{-1}{2\pi f X_s}
  \]  
  (Only valid if `Xs < 0`)

- **Parallel Inductance:**  
  \[
  L_p = \frac{1}{2\pi f |B_p|} \quad \text{(if Bp < 0)}
  \]

- **Parallel Capacitance:**  
  \[
  C_p = \frac{B_p}{2\pi f} \quad \text{(if Bp > 0)}
  \]

---

## 🧪 Use Cases

- Component testing (capacitors, inductors, batteries)
- Sensor or biomedical impedance analysis
- PCB and RF design
- Frequency response and resonance analysis

---

> 📂 **Tip:** Use impedance sweeps over frequency to derive useful equivalent models (series or parallel RLC), and observe transitions from capacitive to inductive behavior.

