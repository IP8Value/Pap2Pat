Here is the drafted patent application following the provided outline:

---

# DESCRIPTION  

## TECHNICAL FIELD  

The present invention relates to battery modeling systems, and more particularly to a method and system for generating an equivalent circuit model of a lithium-sulfur (Li-S) battery that accounts for state-of-charge (SoC) dependencies, temperature variations, and current profile effects. The invention provides a robust parameterization framework suitable for battery management system (BMS) applications in electric vehicles (EVs) and other energy storage systems.  

## BACKGROUND  

Lithium-sulfur batteries represent a promising alternative to conventional lithium-ion batteries due to their high theoretical specific energy and potential cost advantages. However, Li-S batteries exhibit complex electrochemical behavior characterized by multiple voltage plateaus, polysulfide shuttle effects, and significant parameter variations with SoC, temperature, and current. These characteristics make accurate modeling challenging, particularly for real-time BMS applications where computational efficiency is critical.  

Existing equivalent circuit network (ECN) models for lithium-ion batteries are insufficient for Li-S systems because they fail to capture the unique nonlinearities and state dependencies of Li-S chemistry. Prior attempts to model Li-S batteries have focused on fixed-state impedance analysis or overly complex electrochemical models unsuitable for BMS implementation. There remains a need for an operational Li-S battery model that balances accuracy with computational simplicity while accounting for the dynamic parameter variations inherent to Li-S systems.  

## BRIEF SUMMARY OF THE INVENTION  

The present invention provides a novel modeling framework for Li-S batteries based on a behavioral parameterization of equivalent circuit components. The method comprises:  

1. A Thevenin-style equivalent circuit model reparameterized in terms of observable behavioral characteristics rather than physical component values;  
2. A linearization technique that captures short-term parameter variations due to SoC changes during current pulses;  
3. A prediction error minimization (PEM) system identification approach optimized for Li-S parameter extraction;  
4. Temperature-dependent polynomial representations of model parameters with smooth transitions between voltage plateaus;  
5. An interpolation scheme for temperature variations that maintains model differentiability for state estimation applications.  

The resulting model provides accurate voltage prediction across the full operating range of Li-S batteries while maintaining computational efficiency suitable for embedded BMS implementation. The behavioral parameterization enables intuitive constraint application and robust parameter identification from experimental data.  

## DESCRIPTION OF THE EMBODIMENTS  

### Generating the Cell Model Module—Equivalent Circuit Example  

The fundamental cell model comprises a Thevenin equivalent circuit with SoC-dependent parameters. The circuit includes an open-circuit voltage source U<sub>OC</sub>(X) in series with an internal resistance R<sub>0</sub>(X) and a parallel RC network (R<sub>p</sub>(X), C<sub>p</sub>(X)), where X represents the state of charge.  

The behavioral parameterization transforms the conventional component values into intuitive behavioral metrics:  
- Total steady-state resistance R<sub>int</sub> = R<sub>0</sub> + R<sub>p</sub>  
- Dynamic fraction r<sub>p</sub> = R<sub>p</sub>/R<sub>int</sub>  
- Dynamic bandwidth ω<sub>p</sub> = 1/(R<sub>p</sub>C<sub>p</sub>)  

This representation allows direct constraint application on observable behaviors (e.g., limiting ω<sub>p</sub> to physiologically plausible values) and more robust parameter identification. The state equations for the system are:  

dX/dt = -I<sub>L</sub>/Q<sub>cap</sub>  
dU<sub>p</sub>/dt = -ω<sub>p</sub>U<sub>p</sub> + ω<sub>p</sub>r<sub>p</sub>R<sub>int</sub>I<sub>L</sub>  
V<sub>t</sub> = U<sub>OC</sub>(X) - R<sub>int</sub>(1-r<sub>p</sub>)I<sub>L</sub> - U<sub>p</sub>  

where I<sub>L</sub> is load current, Q<sub>cap</sub> is cell capacity, and V<sub>t</sub> is terminal voltage.  

### Generating the Memory Effect Model  

The invention accounts for Li-S memory effects through:  
1. A dual-polynomial representation of U<sub>OC</sub>(X) and R<sub>0</sub>(X) with smooth transitions between high (≈2.35V) and low (≈2.1V) voltage plateaus using a sinusoidal blending function:  

g<sub>m,c</sub>(X) = 0.5[1 + sin(π(X-c)/m)] for c-m/2 < X < c+m/2  

where c is the transition center point and m controls the transition width.  

2. Explicit modeling of the polysulfide shuttle effect through an additional self-discharge current term in the SoC derivative:  

dX/dt = -(I<sub>L</sub> + I<sub>shuttle</sub>)/Q<sub>cap</sub>  

where I<sub>shuttle</sub> is empirically determined as a function of SoC and temperature.  

### Parameterisation by Prediction Error Minimisation (PEM)  

The PEM identification process comprises:  
1. Experimental data collection using mixed-current pulse tests at multiple temperatures (e.g., 20°C, 30°C, 50°C);  
2. Windowed parameter estimation focusing on 300s intervals before and after each current pulse;  
3. Simultaneous identification of all behavioral parameters (R<sub>int</sub>, r<sub>p</sub>, ω<sub>p</sub>, U<sub>OC</sub>) for each pulse;  
4. Polynomial fitting to the SoC-dependent parameters with separate representations for each voltage plateau;  
5. Temperature interpolation of polynomial coefficients through a 3×1 lookup table for each parameter.  

The linearized model form enables efficient PEM implementation while capturing the dominant nonlinear effects through the additional term c'<sub>11</sub> representing the SoC sensitivity of U<sub>OC</sub>. Validation tests demonstrate root-mean-square voltage prediction errors below 32mV across typical operating conditions.  

--- 

The application continues with detailed descriptions of experimental procedures, parameter identification results, and model validation data as required for a complete patent specification. The behavioral parameterization approach and dual-plateau modeling technique represent key novel aspects of the invention.