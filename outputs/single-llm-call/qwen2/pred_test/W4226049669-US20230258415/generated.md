# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to a method and apparatus for enhancing heat transfer and reducing drag in laminar and turbulent fluid flows using liquid-infused surfaces (LIS) with transverse grooves. Specifically, the invention provides a detailed approach to designing and implementing LIS with optimized thermal and flow properties to achieve superior heat transfer performance while minimizing friction losses.

## BACKGROUND

The simultaneous enhancement of heat transfer and reduction of drag in fluid flows has long been a significant challenge in various industrial and engineering applications. Traditional methods often involve the use of rough or modified surfaces to increase the surface area and promote heat exchange. However, these techniques typically result in increased friction, which counteracts the benefits of enhanced heat transfer. Miniaturization, another common approach, increases the surface-to-volume ratio but also leads to higher wall friction and increased pressure drops.

Recent advancements in surface engineering have introduced liquid-infused surfaces (LIS) as a promising solution. LIS utilize a combination of microstructured surfaces and a lubricating liquid to create dynamic interfaces that reduce friction while enhancing heat transfer. The careful design of the surface texture, liquid properties, and thermal conductivities of the solid and liquid phases can significantly influence the overall performance of the system.

However, most existing studies have focused on the friction reduction capabilities of LIS, with limited attention to their heat transfer enhancement properties, especially in the context of transverse grooves. The present invention addresses this gap by providing a comprehensive method for optimizing LIS with transverse grooves to achieve both enhanced heat transfer and reduced drag in both laminar and turbulent flows.

## BRIEF SUMMARY

The invention provides a method and apparatus for enhancing heat transfer and reducing drag in fluid flows using liquid-infused surfaces (LIS) with transverse grooves. The key aspects of the invention include:

1. **Surface Design**: The LIS is designed with transverse grooves that induce recirculation within the grooves, leading to dispersive convection and increased heat transfer.
2. **Material Selection**: The thermal conductivities of the solid and infusing liquid are carefully chosen to optimize heat transfer. Specifically, the thermal conductivity of the solid should be similar to or less than that of the infusing liquid.
3. **Flow Properties**: The method accounts for the effects of Reynolds and Péclet numbers on the dispersive convection and heat transfer enhancement.
4. **Turbulent Flow Considerations**: The invention also provides a framework for evaluating the performance of LIS in turbulent flows, including the impact of dispersive convection on the overall heat transfer and drag reduction.

The invention is particularly useful in applications such as heat exchangers, microprocessors, and other thermal management systems where both efficient heat transfer and low friction are critical.

## DETAILED DESCRIPTION

### Example

#### Introduction

The present invention addresses the challenge of simultaneously enhancing heat transfer and reducing drag in fluid flows by utilizing liquid-infused surfaces (LIS) with transverse grooves. The LIS consists of a microstructured surface infused with a lubricating liquid, creating dynamic interfaces that reduce friction while promoting heat transfer. The invention is particularly effective in both laminar and turbulent flows, making it suitable for a wide range of applications.

#### Governing Equations

The behavior of the fluid and heat transfer in the LIS is governed by the following equations:

1. **Momentum Equation**:
   \[
   \rho \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right) = -\nabla P + \mu \nabla^2 \mathbf{u}
   \]
2. **Continuity Equation**:
   \[
   \nabla \cdot \mathbf{u} = 0
   \]
3. **Energy Equation**:
   \[
   \rho c_p \left( \frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T \right) = \kappa \nabla^2 T
   \]

Where:
- \(\rho\) is the density,
- \(t\) is the time,
- \(\mathbf{u}\) is the fluid velocity,
- \(P\) is the pressure,
- \(\mu\) is the fluid viscosity,
- \(c_p\) is the specific heat capacity,
- \(T\) is the temperature,
- \(\kappa\) is the thermal conductivity.

The equations are valid in the domains occupied by the external fluid and the infusing liquid. In the solid domain, only the energy equation is valid, and the velocity \(\mathbf{u} = 0\).

#### Heat Flux Decomposition

The surface-averaged heat flux \(q\) can be decomposed into different contributions using the Fukagata, Iwamoto, and Kasagi (FIK) identity of the energy equation. The expression for \(q\) is:

\[
q = q_{\text{cond},\infty} + q_{\text{cond},s} + q_{\text{cond},i} + q_{\text{conv},r} + q_{\text{conv},d}
\]

Where:
- \(q_{\text{cond},\infty}\) is the conduction in the external liquid,
- \(q_{\text{cond},s}\) is the conduction in the solid,
- \(q_{\text{cond},i}\) is the conduction in the infusing liquid,
- \(q_{\text{conv},r}\) is the convection from random fluctuations,
- \(q_{\text{conv},d}\) is the convection from dispersive fluctuations.

#### Heat Flux for Varying Solid Conductivity

In laminar flow, the heat flux \(q\) can be expressed as:

\[
q = q_{\text{cond},\infty} + q_{\text{cond},s} + q_{\text{cond},i} + q_{\text{conv},d}
\]

The reference heat flux \(q_0\) for a smooth surface is:

\[
q_0 = \frac{\kappa_s + \kappa_i}{h + 2k} \left( T_u - T_l \right)
\]

Where:
- \(h\) is the height of the channel,
- \(k\) is the depth of the grooves,
- \(T_u\) and \(T_l\) are the temperatures at the upper and lower boundaries, respectively.

The ratio \(q/q_0\) increases with decreasing solid thermal conductivity \(\kappa_s\) relative to the infusing liquid thermal conductivity \(\kappa_i\). For \(\kappa_s \approx \kappa_i\), the dispersive convection \(q_{\text{conv},d}\) plays a crucial role in enhancing the heat flux.

#### Properties of Dispersive Convection

The relative contribution of dispersive convection to the total heat flux, \(q_{\text{conv},d}/q\), depends on the Reynolds and Péclet numbers based on the slip velocity and groove height. The relationship is given by:

\[
\frac{q_{\text{conv},d}}{q} = \frac{0.73k}{h + 2k} \log_{10} \left( \frac{Pe_i}{10} \right)
\]

Where:
- \(Pe_i = Re_i Pr_i\) is the Péclet number based on the slip velocity and groove height,
- \(Re_i = \frac{\rho U_s k}{\mu_i}\) is the Reynolds number,
- \(Pr_i = \frac{c_p \mu_i}{\kappa_i}\) is the Prandtl number.

#### Surface Nusselt Number

The heat transfer through the LIS can be quantified by a surface Nusselt number \(Nu_i\):

\[
Nu_i = \frac{q_{\text{conv},d}}{q} \left( \frac{h + 2k}{k} \right)
\]

The upper limit of \(q_{\text{conv},d}/q\) is given by:

\[
\frac{q_{\text{conv},d}}{q} \leq \frac{k}{h + 2k}
\]

#### Turbulent Flow Considerations

In turbulent flows, the dispersive convection \(q_{\text{conv},d}\) is amplified by the random fluctuations in the bulk flow. The change in heat flux can be expressed as:

\[
\frac{q}{q_0} = 1 + \frac{Nu_0 q_{\text{conv},d}}{q_0} + \frac{q - q_0}{q_0}
\]

Where:
- \(Nu_0\) is the Nusselt number of the smooth-wall flow.

The dispersive convection \(q_{\text{conv},d}\) can be predicted using the relationship:

\[
\frac{q_{\text{conv},d}}{q_0} = \frac{0.73k}{h + 2k} \log_{10} \left( \frac{Pe_i}{10} \right)
\]

#### Heat Flux to Drag Ratio

The heat transfer efficiency of the system can be measured by the heat flux to drag ratio, \(2St/C_f\), where \(St\) is the Stanton number and \(C_f\) is the friction coefficient. For the current set-up, the ratio exceeds unity, indicating an increase in heat transfer efficiency.

#### Conclusion

The present invention provides a method and apparatus for enhancing heat transfer and reducing drag in fluid flows using liquid-infused surfaces (LIS) with transverse grooves. By carefully designing the surface texture, selecting appropriate materials, and accounting for flow properties, the invention achieves superior heat transfer performance while minimizing friction losses. This makes the invention particularly useful in applications such as heat exchangers, microprocessors, and other thermal management systems.