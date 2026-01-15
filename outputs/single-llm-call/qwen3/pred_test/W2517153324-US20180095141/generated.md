# DESCRIPTION

## TECHNICAL FIELD

- relate to methods and apparatuses

The present invention relates to methods and apparatuses for estimating the state of charge and state of health of lithium-sulfur electrochemical cells under dynamic operational conditions, particularly in the context of electric vehicle propulsion systems. The invention encompasses a computational framework that integrates a physics-informed equivalent circuit model with a memory effect module to account for the unique electrochemical behavior of lithium-sulfur chemistry, including polysulfide shuttle dynamics, capacity fading due to cycling history, and temperature-dependent voltage hysteresis. The apparatus is configured to receive real-time measurements of terminal voltage, current, and cell temperature, and to generate an estimate of the cell’s internal state through an iterative feedback mechanism that minimizes prediction error between observed and modeled outputs. The invention further extends to computer-readable media storing executable instructions for implementing the estimation algorithm, and to battery management systems that utilize the estimated state of charge to determine vehicle range, optimize charging protocols, and plan energy-efficient routes under variable environmental and load conditions.

## BACKGROUND

- motivate need for SOC determination
- introduce difficulty of measuring residual energy
- define state of charge and state of discharge
- provide SOC formula
- explain initial SOC setting
- introduce Lithium Sulfur cell challenges
- describe OCV curve of Lithium Sulfur cell
- discuss limitations of resistance and temperature measurements
- discuss limitations of coulomb counting

Accurate determination of the state of charge in electrochemical energy storage systems is critical for ensuring safe, efficient, and reliable operation, particularly in applications such as electric vehicles where range anxiety directly impacts user adoption. Unlike conventional lithium-ion systems, lithium-sulfur cells exhibit a highly non-linear and history-dependent voltage profile that renders traditional state-of-charge estimation techniques inadequate. The state of charge is defined as the ratio of the remaining usable capacity to the cell’s nominal capacity at a reference condition, while the state of discharge is its complement, representing the fraction of capacity already extracted. A common formula for state of charge is given by the integral of current over time normalized by the cell’s rated capacity, adjusted for coulombic efficiency. However, initial state of charge settings are often assumed based on factory calibration or open-circuit voltage readings, which are unreliable in lithium-sulfur systems due to prolonged relaxation dynamics and self-discharge phenomena. Lithium-sulfur cells face intrinsic challenges arising from the multi-step reduction of elemental sulfur through soluble polysulfide intermediates to insoluble lithium sulfide, resulting in a dual-plateau discharge curve with a high-voltage plateau near 2.35 V and a low-voltage plateau near 2.1 V. The open-circuit voltage in the low plateau consistently returns to approximately 2.15 V after rest, regardless of prior state of charge, rendering it unsuitable as a direct indicator of remaining energy. Furthermore, internal resistance measurements are confounded by the formation of insulating lithium sulfide films and the variable viscosity of polysulfide-rich electrolytes, which cause transient resistance spikes that do not correlate linearly with state of charge. Temperature measurements alone fail to capture the complex interplay between reaction kinetics, diffusion rates, and polysulfide solubility. Coulomb counting, while theoretically straightforward, accumulates error over time due to unmeasured leakage currents, parasitic shuttle reactions, and irreversible capacity loss from active material precipitation, making it unsuitable for long-term state estimation without frequent recalibration. These limitations collectively impede the deployment of lithium-sulfur batteries in real-world applications unless a robust, adaptive, and chemistry-aware estimation framework is employed.

## BRIEF SUMMARY OF THE INVENTION

- introduce Lithium Sulfur chemistry limitations
- describe dynamic use conditions
- explain Qt parameter variation
- define state of health (SoH)
- motivate Lithium Sulfur batteries
- describe pouch cell format
- introduce memory effect problem
- describe invention context
- introduce apparatus for modelling SOC
- describe cell model module
- describe memory effect module
- explain apparatus configuration
- describe operational condition of cell
- introduce equivalent circuit network model
- describe parameterised physics-based cell model
- introduce parameter value resource
- describe memory model
- explain reaction rates
- introduce simplified physical model
- describe memory effect module configuration
- introduce apparatus for estimating SOC
- describe cell operational condition monitor module
- describe state estimator module
- describe state of charge estimator module
- introduce electrochemical cell chemistry
- describe state of health estimation
- introduce iterative feedback loop
- describe kalman-type filter
- introduce prediction error minimisation technique
- describe cell operational condition measurement means
- introduce invention
- define parameter values
- motivate battery management system
- describe apparatus for estimating range
- describe apparatus for planning route
- describe computer readable medium
- motivate method for generating model
- describe method for generating model
- describe alternative cell models
- describe equivalent circuit network model
- describe operational condition of cell
- describe generating data for model
- describe predicting terminal voltage
- describe controlled testing of cells
- describe applying current pulses
- describe identifying parameters of model
- describe using open circuit voltage
- describe using instantaneous voltage drop
- describe using gradual voltage drop
- describe refining parameter values
- describe storing parameter values
- describe fitting parameter values to functions
- motivate method for generating memory model
- describe method for generating memory model
- describe establishing rules for reactant species
- describe parameterising reaction rates
- describe identifying parameterised values
- describe simplified physical model
- motivate method for estimating state of charge
- describe method for estimating state of charge
- describe estimating internal state of cell
- describe estimating usable capacity of cell
- describe estimating range of electric vehicle

The invention addresses the fundamental limitations of lithium-sulfur electrochemical cells by introducing a novel apparatus and method for estimating state of charge and state of health through a hybrid modeling architecture that combines a parameterized physics-based equivalent circuit network with a memory effect module that captures the cumulative influence of prior operational history. Lithium-sulfur chemistry, while offering high theoretical specific energy, suffers from pronounced capacity fade, voltage hysteresis, and self-discharge mechanisms driven by the reversible dissolution and shuttle of intermediate polysulfide species. Under dynamic use conditions such as those encountered in electric vehicles, the usable capacity of the cell, denoted Qt, varies non-linearly with discharge rate, temperature, and cycle count, necessitating a model that evolves with operational history rather than assuming static parameters. State of health is defined as the ratio of the current maximum usable capacity to the initial rated capacity, accounting for irreversible degradation from polysulfide loss and anode passivation. The invention is contextually situated within the deployment of lithium-sulfur pouch cells in automotive applications, where compact form factor and high energy density are paramount. The memory effect problem arises from the fact that the cell’s voltage response to a given current profile depends not only on its immediate state but also on the sequence and magnitude of prior discharges, leading to predictable but non-reversible shifts in the open-circuit voltage curve and discharge plateau transitions. The apparatus comprises a cell model module that implements a simplified Thevenin-type equivalent circuit network with a single resistor-capacitor pair, parameterized as a function of state of charge, temperature, and cycle number. A parameter value resource stores fitted polynomial functions that describe the variation of open-circuit voltage, ohmic resistance, charge transfer resistance, and double-layer capacitance across the discharge range and temperature spectrum. The memory effect module is configured to track the cumulative history of polysulfide concentration gradients and anode film thickness, adjusting the effective reaction rates and diffusion coefficients in real time. A simplified physical model is embedded within the memory module to approximate the kinetics of lithium sulfide precipitation and re-dissolution based on empirical rules derived from experimental observations. The apparatus further includes a cell operational condition monitor module that continuously measures terminal voltage, current, and temperature, feeding these inputs into a state estimator module that employs a Kalman-type filter to recursively minimize the prediction error between measured and modeled terminal voltage. The state of charge estimator module outputs a refined estimate of the cell’s internal state, which is then used to compute the state of health by comparing the current usable capacity to a baseline value established during initial calibration. The invention introduces an iterative feedback loop wherein the prediction error is used to adaptively update the parameter values of the cell model and the memory model, ensuring continuous alignment with real-world degradation patterns. The parameter values are defined as coefficients of piecewise polynomial functions that describe the non-linear dependence of circuit elements on state of charge and temperature, with transition points between voltage plateaus dynamically adjusted based on historical discharge profiles. The invention motivates the integration of this estimation framework into a battery management system that enables accurate range prediction for electric vehicles by correlating the estimated state of charge and state of health with known power demand profiles. The apparatus may be implemented on a computer-readable medium containing executable instructions for performing the estimation algorithm, and may be deployed in embedded control units or cloud-based fleet management systems. The method for generating the cell model involves subjecting cells to controlled current pulses across a range of temperatures and discharge rates, measuring the resulting voltage transients, and identifying model parameters using prediction error minimization techniques. The method for generating the memory model establishes rules governing the evolution of reactant species concentrations based on discharge depth and rest duration, parameterizes reaction rates as functions of temperature and polysulfide density, and identifies model parameters through curve fitting of capacity retention curves over hundreds of cycles. The method for estimating state of charge involves estimating the internal state of the cell by solving a state-space representation of the equivalent circuit, incorporating the memory-adjusted parameters, and using the Kalman filter to correct for model inaccuracies. The usable capacity of the cell is estimated by integrating the current over time while dynamically adjusting the capacity scaling factor based on the memory model’s output, and the range of the electric vehicle is computed by multiplying the estimated usable capacity by the average energy consumption rate derived from driving patterns and environmental conditions.

## DESCRIPTION OF THE EMBODIMENTS

- introduce Lithium Sulfur cells
- describe memory effect in Lithium Sulfur cells
- explain limitations of internal resistance for SOC estimation
- introduce apparatus for modelling and estimating SOC and SOH
- describe cumulative history data collection
- introduce terminal voltage estimation method
- describe cell operational condition measurement means
- explain internal resistance measurement
- introduce SOC model
- describe cell model module
- explain parameter value resource
- introduce memory effect module
- describe memory model
- explain reaction rates parameterisation
- introduce cell state estimator
- describe SOC estimation method
- explain alternative embodiments
- introduce apparatus for creating SOC model
- describe cell state estimator implementation
- explain SOC model implementation
- introduce battery management system
- describe energy system controller
- explain plural cells implementation
- introduce specific implementations of apparatus

Lithium-sulfur cells operate through a complex conversion mechanism involving the reduction of sulfur to lithium sulfide via soluble polysulfide intermediates, which leads to a distinctive voltage profile characterized by two distinct plateaus separated by a transition region of increasing internal resistance. The memory effect in these cells manifests as a persistent shift in the open-circuit voltage curve and a reduction in usable capacity following deep discharges or prolonged rest periods, attributable to the irreversible accumulation of lithium sulfide on the anode and the redistribution of polysulfide species within the electrolyte. Internal resistance measurements alone are insufficient for state of charge estimation because they reflect transient phenomena such as electrolyte viscosity changes and film formation rather than the true electrochemical state of the cell. The apparatus for modeling and estimating state of charge and state of health comprises a modular architecture that decouples the dynamic electrical response from the cumulative chemical degradation, enabling accurate estimation even under rapidly changing load and temperature conditions. Cumulative history data collection is performed by logging the time-integrated current, rest durations, temperature extremes, and discharge depth over multiple cycles, which are then used to update the memory model’s internal state variables. The terminal voltage estimation method employs a state-space representation of the equivalent circuit, where the predicted voltage is computed as the sum of the open-circuit voltage, the ohmic drop, and the RC transient response, all modulated by the memory-adjusted parameters. Cell operational condition measurement means include high-precision current sensors, thermocouples, and voltage acquisition circuits synchronized to a high-resolution sampling clock. Internal resistance is measured as the instantaneous voltage drop following a short current pulse, but this value is not used directly for state estimation; instead, it serves as a validation metric for the model’s internal resistance parameter. The state of charge model is implemented as a parameterized physics-based equivalent circuit network, where each component value is expressed as a function of state of charge, temperature, and cycle count. The parameter value resource is a non-volatile memory store containing pre-fitted polynomial coefficients for open-circuit voltage, ohmic resistance, charge transfer resistance, and capacitance, each indexed by temperature and discharge plateau region. The memory effect module implements a simplified physical model that tracks the evolution of lithium sulfide film thickness and polysulfide concentration gradients using empirical rules derived from experimental data, adjusting the reaction rate constants and diffusion coefficients in real time. Reaction rates are parameterized as Arrhenius-type functions modulated by the local concentration of soluble polysulfides and the surface coverage of the anode. The cell state estimator is implemented as a Kalman-type filter that predicts the next state based on the current model and corrects the prediction using the measured terminal voltage, thereby compensating for model inaccuracies and measurement noise. The state of charge estimation method iteratively updates the state vector comprising state of charge, capacitor voltage, and memory state variables, using a prediction error minimization criterion that minimizes the root mean square deviation between predicted and measured voltage over a moving time window. Alternative embodiments include the use of multiple RC elements to capture slower relaxation dynamics, or the integration of electrochemical impedance spectroscopy data for offline model refinement. The apparatus for creating the state of charge model is implemented as a laboratory system that applies controlled current pulses across a range of temperatures and discharge rates, identifies model parameters via non-linear least squares fitting, and generates polynomial functions that interpolate between measured data points. The cell state estimator is implemented as a real-time algorithm running on a microcontroller with fixed-point arithmetic, optimized for low computational overhead. The state of charge model is implemented as a set of lookup tables and polynomial evaluators that compute the model parameters on-the-fly based on the current state of charge and temperature. The battery management system incorporates the state of charge and state of health estimates to regulate charging current, prevent over-discharge, and trigger maintenance alerts. The energy system controller uses the estimated range to dynamically adjust climate control, regenerative braking intensity, and power delivery to ancillary systems. Plural cells are implemented in series or parallel configurations, with each cell monitored independently and the overall pack state computed as a weighted average based on individual health metrics. Specific implementations of the apparatus include embedded battery management units integrated into electric vehicle battery packs, cloud-based fleet analytics platforms, and diagnostic tools for battery recycling and second-life applications.

### Generating the Cell Model Module—Equivalent Circuit Example

- introduce equivalent circuit model
- describe test data generation
- explain current load application
- describe terminal voltage measurement
- introduce equivalent circuit network model
- describe model structure selection
- explain parameterisation process
- introduce open circuit voltage calculation
- describe ohmic resistance calculation
- explain diffuse resistance calculation
- introduce fitting procedure
- describe non-linear least squares technique
- explain look up table creation
- describe curve fitting
- introduce fitted polynomials
- explain model validation

The equivalent circuit model employed in the invention is a single resistor-capacitor Thevenin configuration, chosen for its balance between computational efficiency and predictive accuracy under dynamic conditions. Test data generation involves applying a sequence of controlled current pulses—ranging from low-rate C/10 to high-rate 2C—to lithium-sulfur pouch cells held at temperatures between 10°C and 50°C, with rest periods between pulses to allow voltage relaxation. Current load application is performed using a programmable battery tester capable of precise current sourcing and sinking, while terminal voltage is measured with a high-precision data acquisition system sampling at 100 Hz. The equivalent circuit network model is selected based on its ability to capture the dominant transient response without introducing unnecessary complexity, and its structure is validated against impedance spectroscopy data collected at multiple states of charge. The parameterisation process begins with the identification of the open-circuit voltage as a function of state of charge, computed from the voltage measured after extended rest periods, excluding periods immediately following high-current pulses. Ohmic resistance is calculated from the instantaneous voltage drop at the onset of each current pulse, while diffuse resistance and capacitance are derived from the exponential decay of voltage following the pulse termination. The fitting procedure employs a non-linear least squares technique to minimize the sum of squared errors between the measured terminal voltage and the model output over the entire discharge profile, with constraints applied to ensure physical plausibility of the parameters. A lookup table is created for each parameter across discrete state of charge intervals and temperature points, and curve fitting is performed using piecewise polynomial functions to interpolate between these points with continuous first derivatives. Fitted polynomials are generated for each parameter independently, with separate functions defined for the high and low voltage plateaus to account for the distinct electrochemical mechanisms in each region. Model validation is conducted by comparing the model’s predicted voltage against an independent dataset obtained under a realistic drive cycle, such as the NEDC profile, with the root mean square error serving as the primary metric of accuracy.

### Generating the Memory Effect Model

- introduce Lithium Sulfur memory effect
- motivate Qt variation
- describe capacity variation experiment
- illustrate capacity variation results
- define memory effect
- introduce memory model
- describe memory model functionality
- expand memory model for degradation
- define state of health
- describe memory model operation
- generate LiS model rules
- calculate cell voltage
- predict reaction rates
- describe simplified memory model
- adjust ECN model for memory effect
- illustrate adjustment mechanism
- show modelled discharge curves
- illustrate capacity loss
- parametrise simplified memory model
- describe complex memory model variation

The lithium-sulfur memory effect arises from the irreversible accumulation of lithium sulfide on the anode surface and the redistribution of polysulfide species during repeated cycling, leading to a progressive and non-recoverable reduction in usable capacity, denoted Qt, even under identical discharge conditions. Capacity variation experiments involve subjecting cells to repeated discharge cycles at fixed current rates and temperatures, measuring the remaining capacity after each cycle, and observing a consistent decline that correlates with the depth of discharge and the duration of rest periods. The memory effect is defined as the persistent deviation in voltage response and capacity retention that cannot be explained by state of charge alone but is instead a function of the cell’s operational history. The memory model is a computational module that tracks the cumulative history of polysulfide concentration gradients and anode film thickness, updating internal state variables after each discharge or rest event. The functionality of the memory model is to adjust the effective reaction rates and diffusion coefficients in the equivalent circuit model based on the accumulated history, thereby modifying the predicted open-circuit voltage and charge transfer resistance. The memory model is expanded to account for degradation by incorporating a state of health variable that scales the maximum usable capacity downward over time, with the rate of decline governed by empirical rules derived from accelerated aging tests. State of health is defined as the ratio of the current maximum capacity to the initial rated capacity, and is updated incrementally based on the deviation between predicted and measured capacity after each full discharge. The memory model operates by applying a set of heuristic rules that link discharge depth, rest duration, and temperature to the rate of lithium sulfide precipitation and polysulfide loss, with each rule calibrated against experimental data. Cell voltage is calculated by combining the open-circuit voltage, adjusted by the memory state, with the resistive and capacitive components of the equivalent circuit. Reaction rates are predicted using an Arrhenius formulation modulated by the local concentration of soluble polysulfides and the surface coverage of the anode. The simplified memory model represents the cumulative degradation as a single state variable that evolves according to a first-order differential equation driven by the discharge depth and rest time. The equivalent circuit model is adjusted by scaling the charge transfer resistance and the open-circuit voltage curve based on the memory state, effectively shifting the voltage profile to reflect the reduced capacity and altered kinetics. The adjustment mechanism is illustrated by overlaying modelled discharge curves before and after memory accumulation, showing a clear downward shift in voltage and a reduction in the length of the high plateau. Capacity loss is illustrated by plotting the remaining capacity over hundreds of cycles, demonstrating a logarithmic decay pattern that is accurately captured by the memory model. The simplified memory model is parametrized using a single decay constant and a scaling factor determined through nonlinear regression of capacity retention data. Complex variations of the memory model include additional state variables tracking the concentration of specific polysulfide species and the spatial distribution of lithium sulfide deposits, enabling higher fidelity predictions at the cost of increased computational load.

### Parameterisation by Prediction Error Minimisation (PEM)

- introduce PEM method
- describe advantages of PEM
- outline PEM procedure
- describe model structure selection
- illustrate equivalent circuit model
- describe model structure selection criteria
- describe fitting parameters to the model
- define prediction error
- describe iterative minimization procedure
- define fitness function
- describe identification error minimisation
- define RMSE criterion
- illustrate RMSE values
- describe real-time state of charge estimation methods
- illustrate state of charge estimation method
- describe current, voltage, and temperature measurement
- describe updating state vector
- describe using equivalent circuit model
- describe predicting model parameters
- describe estimating state of health
- describe outputting measurable parameters
- describe feedback loop
- describe increasing accuracy
- describe deploying memory-aware model
- describe generating model from experimental data
- describe implementing equivalent circuit model
- describe using Kalman-type filter
- illustrate real-time SOC/SOH estimation
- describe receiving measured values
- describe updating state vector
- describe predicting internal cell state
- describe correcting predictions
- describe adapting ECN model
- describe algorithm architecture
- describe deployment architectures
- describe selecting prediction horizon

The prediction error minimization method is employed to identify the parameters of the equivalent circuit model by iteratively adjusting them to minimize the difference between the measured terminal voltage and the voltage predicted by the model over a defined time window. The advantages of PEM include its robustness to noise, its ability to handle non-linear and time-varying systems, and its compatibility with real-time implementation on embedded hardware. The PEM procedure begins with the selection of a model structure—here, a single RC Thevenin network—based on its ability to capture the dominant dynamics without excessive complexity. The equivalent circuit model is illustrated as a voltage source representing open-circuit voltage in series with an ohmic resistor and a parallel resistor-capacitor branch. Model structure selection criteria prioritize computational efficiency, physical interpretability, and predictive accuracy under transient conditions. Fitting parameters to the model involves defining a cost function that quantifies the prediction error as the sum of squared differences between measured and predicted voltage over a moving window of data. The prediction error is defined as the residual between the actual terminal voltage and the model’s output, computed at each sampling instant. The iterative minimization procedure employs a gradient-based optimization algorithm that adjusts the model parameters in the direction that reduces the cost function, with convergence determined by a threshold on the change in parameter values between iterations. The fitness function is defined as the inverse of the root mean square error, such that higher fitness corresponds to lower prediction error. Identification error minimisation is achieved by constraining the parameter search space to physically plausible ranges and by initializing the optimization with values derived from prior experimental data. The RMSE criterion is used to quantify model performance, with values below 35 mV indicating acceptable accuracy for real-time applications. Real-time state of charge estimation methods rely on a Kalman-type filter that predicts the next state of the cell using the equivalent circuit model and corrects the prediction using the measured voltage. The state of charge estimation method is illustrated as a feedback loop where current, voltage, and temperature measurements are sampled at 10 Hz and fed into the state estimator. The state vector, comprising state of charge, capacitor voltage, and memory state, is updated at each time step using the model’s state transition equations. The equivalent circuit model is used to predict the terminal voltage given the current state and input current, and the prediction error is used to adjust the state vector via a Kalman gain. Model parameters are predicted dynamically by interpolating between pre-fitted polynomial functions based on the current state of charge and temperature. State of health is estimated by comparing the current maximum capacity, inferred from the memory model, to the initial rated capacity. Measurable parameters such as terminal voltage, current, and temperature are continuously output to the battery management system. The feedback loop increases accuracy over time by adapting the memory model’s parameters in response to persistent prediction errors, allowing the system to self-calibrate under varying usage patterns. The memory-aware model is deployed in embedded battery management units, where it operates continuously during vehicle operation. The model is generated from experimental data collected under controlled conditions and validated against real-world drive cycles. The equivalent circuit model is implemented as a set of polynomial evaluators and state transition functions running on a microcontroller. The Kalman-type filter is implemented using fixed-point arithmetic to ensure real-time performance. Real-time state of charge and state of health estimation is illustrated as a continuous output stream updated every 100 milliseconds, with the estimated values used to drive the vehicle’s range display and energy management system. Measured values are received from sensors via a high-speed analog-to-digital converter, the state vector is updated using the model’s discrete-time equations, the internal cell state is predicted using the equivalent circuit model, and predictions are corrected using the Kalman gain. The ECN model is adapted by scaling its parameters based on the memory state, ensuring that the model evolves with the cell’s degradation. The algorithm architecture consists of three modules: the cell model, the memory model, and the state estimator, each operating in sequence within a fixed-time loop. Deployment architectures include on-board battery management units, cloud-based fleet analytics platforms, and diagnostic tools for end-of-life battery assessment. The prediction horizon is selected to balance computational load and estimation accuracy, with a horizon of 30 seconds proven optimal for capturing transient dynamics without introducing excessive latency.