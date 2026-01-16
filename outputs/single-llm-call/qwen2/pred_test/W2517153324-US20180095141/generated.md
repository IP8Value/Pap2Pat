# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to a method and system for modeling the behavior of lithium-sulfur (Li-S) batteries, particularly for use in battery management systems (BMS) in electric vehicles (EVs). The invention specifically addresses the development of a simplified equivalent circuit network (ECN) model that can accurately predict the voltage response, capacity, and power capability of Li-S batteries under various operating conditions, including different temperatures and current profiles.

## BACKGROUND

Electric vehicles (EVs) are increasingly gaining popularity due to their environmental benefits and reduced operating costs. However, one of the primary challenges in the widespread adoption of EVs is range anxiety, which is closely tied to the performance and reliability of the energy storage systems used in these vehicles. Lithium-sulfur (Li-S) batteries offer a promising alternative to traditional lithium-ion (Li-ion) batteries due to their high theoretical specific energy and potential for lower cost and safer operation. Despite these advantages, Li-S batteries face significant challenges, including rapid degradation and high self-discharge rates, which complicate their operational modeling and control.

Operational models and online diagnostic tools capable of predicting and controlling the performance of Li-S batteries in real-time are crucial for their successful integration into EVs. However, existing models for Li-S batteries are either too complex for practical implementation in BMS or lack the accuracy required for reliable performance prediction. Equivalent circuit network (ECN) models, which have been successfully used for Li-ion batteries, offer a balance between simplicity and accuracy. These models simulate the transient behavior of a battery using a circuit of electrical components, such as resistors, capacitors, and voltage sources.

Despite the potential of ECN models, adapting them for Li-S batteries presents several challenges. The unique chemistry of Li-S batteries, involving conversion reactions and the formation of polysulfides, results in complex behavior that is not well-represented by conventional ECN models. Additionally, the temperature and current dependencies of Li-S batteries are more pronounced and less predictable compared to Li-ion batteries. Therefore, there is a need for a robust and accurate ECN model specifically designed for Li-S batteries that can be efficiently implemented in BMS.

## BRIEF SUMMARY OF THE INVENTION

The present invention provides a method and system for developing a simplified equivalent circuit network (ECN) model for lithium-sulfur (Li-S) batteries suitable for use in battery management systems (BMS) of electric vehicles (EVs). The model is designed to accurately predict the voltage response, capacity, and power capability of Li-S batteries under various operating conditions, including different temperatures and current profiles.

The invention includes the following key aspects:
1. **Generating the Cell Model Module—Equivalent Circuit Example**: The method involves creating a Thevenin equivalent circuit model with parameters that depend on the state of charge (SoC) of the battery. The model includes an open circuit voltage (OCV) source, an internal resistance (R0), and a parallel resistor-capacitor (RC) pair to represent the dynamic behavior of the battery.
2. **Generating the Memory Effect Model**: The method accounts for the memory effect in Li-S batteries, which is characterized by the difference in OCV before and after a current pulse. This is achieved by incorporating a behavioral parameterization that captures the short-term changes in dynamic behavior due to changes in SoC.
3. **Parameterisation by Prediction Error Minimisation (PEM)**: The parameters of the ECN model are identified using a robust parameter estimation technique based on prediction error minimization (PEM). The method involves fitting the model to pulse discharge data collected at different temperatures and current rates. The PEM algorithm minimizes the prediction error between the measured data and the model predictions, ensuring accurate parameter identification.

The invention further includes a validation step where the model is tested against a realistic current profile, such as the New European Driving Cycle (NEDC), to ensure its accuracy and applicability in real-world scenarios. The model demonstrates a low root mean square error (RMSE) and accurately predicts the transient voltage behavior of Li-S batteries throughout the discharge range.

## DESCRIPTION OF THE EMBODIMENTS

### Generating the Cell Model Module—Equivalent Circuit Example

The cell model module is based on a Thevenin equivalent circuit, which is a simplified representation of the battery's electrical behavior. The circuit includes an open circuit voltage (OCV) source, an internal resistance (R0), and a parallel resistor-capacitor (RC) pair to capture the dynamic behavior of the battery. The OCV is a function of the state of charge (SoC), and the internal resistance and RC parameters are also dependent on the SoC.

The basic equations for the Thevenin model are as follows:
\[ V_{\text{out}} = V_{\text{OC}}(X) - R_0(X) \cdot I_L - R_p(X) \cdot I_L + \frac{U_p(X)}{C_p(X)} \]
where:
- \( V_{\text{out}} \) is the terminal voltage of the battery.
- \( V_{\text{OC}}(X) \) is the open circuit voltage, which is a function of the state of charge \( X \).
- \( R_0(X) \) is the internal resistance, which is a function of the state of charge \( X \).
- \( R_p(X) \) is the resistance of the RC pair, which is a function of the state of charge \( X \).
- \( C_p(X) \) is the capacitance of the RC pair, which is a function of the state of charge \( X \).
- \( I_L \) is the load current.
- \( U_p(X) \) is the voltage across the capacitor in the RC pair.

### Generating the Memory Effect Model

The memory effect in Li-S batteries is characterized by the difference in OCV before and after a current pulse. To account for this effect, the model is reparameterized in terms of behavioral variables that capture the short-term changes in dynamic behavior due to changes in SoC. The behavioral variables include the dynamic bandwidth \( U_p \), the total steady-state resistance \( R_{\text{int}} \), and the dynamic fraction \( r_p \).

The reparameterized model equations are:
\[ V_{\text{out}} = V_{\text{OC}}(X) - R_{\text{int}}(X) \cdot I_L - r_p(X) \cdot R_{\text{int}}(X) \cdot I_L + \frac{U_p(X)}{C_p(X)} \]
where:
- \( V_{\text{OC}}(X) \) is the open circuit voltage, which is a function of the state of charge \( X \).
- \( R_{\text{int}}(X) \) is the total steady-state resistance, which is a function of the state of charge \( X \).
- \( r_p(X) \) is the dynamic fraction, which is a function of the state of charge \( X \).
- \( U_p(X) \) is the voltage across the capacitor in the RC pair, which is a function of the state of charge \( X \).

### Parameterisation by Prediction Error Minimisation (PEM)

The parameters of the ECN model are identified using a robust parameter estimation technique based on prediction error minimization (PEM). The method involves fitting the model to pulse discharge data collected at different temperatures and current rates. The PEM algorithm minimizes the prediction error between the measured data and the model predictions, ensuring accurate parameter identification.

The key steps in the parameter identification process are:
1. **Definition of Operating Point**: Define an operating point for the system, which includes the state of charge \( X \) and the capacitor voltage \( U_p \). The nominal input is the current \( I_L \), and the nominal output is the terminal voltage \( V_{\text{out}} \).
2. **Linear State-Space Representation**: Form a linearized model using the state vector \( \mathbf{x} = [X, U_p]^T \). The linearized model is represented by the state-space equations:
   \[ \dot{\mathbf{x}} = \mathbf{A} \mathbf{x} + \mathbf{b} I_L \]
   \[ V_{\text{out}} = \mathbf{c}^T \mathbf{x} + d I_L \]
   where:
   - \( \mathbf{A} \) is the state matrix.
   - \( \mathbf{b} \) is the input matrix.
   - \( \mathbf{c} \) is the output matrix.
   - \( d \) is the direct transmission term.
3. **Minimization of Prediction Error**: Use numerical optimization to minimize the cost function \( V_N(\mathbf{q}) \), which is a weighted norm of the prediction error. The cost function is defined as:
   \[ V_N(\mathbf{q}) = \sum_{i=1}^{N} \left( V_{\text{out},i} - \hat{V}_{\text{out},i}(\mathbf{q}) \right)^2 \]
   where:
   - \( V_{\text{out},i} \) is the measured terminal voltage at time \( i \).
   - \( \hat{V}_{\text{out},i}(\mathbf{q}) \) is the model-predicted terminal voltage at time \( i \).
   - \( \mathbf{q} \) is the parameter vector.

The identified parameters are then used to create a simplified battery model with polynomial functions for the parameters, which are interpolated for different temperatures. The model is validated against a realistic current profile, such as the New European Driving Cycle (NEDC), to ensure its accuracy and applicability in real-world scenarios.

The invention provides a robust and accurate ECN model for Li-S batteries that can be efficiently implemented in BMS, enabling better control and prediction of battery performance in electric vehicles.