# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to electric motors, particularly to high-power, high-reliability electric motors for use in electric vertical takeoff and landing (eVTOL) aircraft. More specifically, the invention pertains to an integrated magnetics, insulation, and cooling architecture (MAGICA) designed to enhance heat transfer from the motor coils to the heatsink, thereby preventing excessive temperatures and ensuring reliable operation under demanding conditions.

## BACKGROUND OF THE INVENTION

Electric vertical takeoff and landing (eVTOL) aircraft require compact, high-power, and high-reliability electric motors and drives. The power demand on the electric aircraft propulsion (EAP) system during take-off and landing can be significantly higher—up to 3-10 times the cruise power—compared to other flight phases. This increased power demand results in higher current densities, leading to a rapid rise in temperature within the conducting regions of the motor. Excessive temperatures can pose a significant threat to the motor's insulation, potentially causing insulation failure and compromising the overall reliability of the system.

Conventional electric motors often struggle to manage the thermal challenges associated with high-power operations, especially in eVTOL applications. Traditional cooling methods, such as radial fin heat sinks, may not be sufficient to dissipate the heat generated during high-power phases. Additionally, the electrical insulation in these motors must be robust enough to withstand the high voltages and currents involved, further complicating the design.

To address these issues, the present invention introduces a novel integrated magnetics, insulation, and cooling architecture (MAGICA) for electric motors. This architecture is specifically designed to improve thermal management and electrical insulation, making it particularly suitable for the demanding operating profiles of eVTOL aircraft.

## SUMMARY OF THE INVENTION

The present invention provides an integrated magnetics, insulation, and cooling architecture (MAGICA) for electric motors, particularly for use in electric vertical takeoff and landing (eVTOL) aircraft. The invention includes a slotless permanent magnet synchronous machine (PMSM) outrunner with an enhanced cooling system that effectively manages heat dissipation and ensures reliable operation under high-power conditions.

Key features of the invention include:
1. **Ceramic Coil Holder**: A ceramic coil holder is used to house the form-wound Litz conductors. This holder introduces an additional side-wall parallel heat path from the windings to the heatsink, enhancing thermal performance. The ceramic material also improves line-to-ground and line-to-line insulation, reducing the risk of insulation failure and partial discharge activity.
2. **Interleaved Copper Sheets**: Thin copper sheets are periodically inserted within the iron lamination stack to provide a thermal shunt from the coils to the heatsink. This reduces the thermal impedance and helps in more efficient heat dissipation.
3. **Additively Manufactured Heatsink**: An additively manufactured heatsink is used to improve air-side heat transfer over the baseline radial fin heat sink through increased surface area. This further enhances the thermal performance of the motor.

The invention is particularly beneficial for eVTOL applications where the power demand during take-off and landing is significantly higher than during cruise. By effectively managing heat dissipation and ensuring robust electrical insulation, the invention enhances the reliability and performance of electric motors in these demanding conditions.

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS

### Field of Application

The present invention is particularly suited for electric motors used in electric vertical takeoff and landing (eVTOL) aircraft. These aircraft require motors that can handle high power demands during take-off and landing, which can be up to 3-10 times the cruise power. The invention addresses the critical issues of thermal management and electrical insulation, ensuring reliable operation under these demanding conditions.

### Design of the Slotless PMSM

The preferred embodiment of the invention is a 300 kW slotless permanent magnet synchronous machine (PMSM) outrunner. The motor consists of an outer rotor Halbach permanent magnet (PM) array, a shell, core lamination, a heatsink, and a slotless 3-phase armature structure. The slotless design allows for high electrical loading by extending the armature cross-section into the area traditionally occupied by iron teeth, making it suitable for the operating profiles of eVTOL aircraft.

### Cooling Architecture

#### Ceramic Coil Holder

One of the critical features of the invention is the ceramic coil holder. This holder is designed to house the form-wound Litz conductors and introduces an additional side-wall parallel heat path from the windings to the heatsink. The ceramic material used in the holder has high thermal conductivity and excellent electrical insulation properties. This dual functionality enhances the thermal performance of the motor while reducing the risk of insulation failure and partial discharge activity.

The dimensions of the ceramic coil holder are carefully optimized to balance thermal performance and electromagnetic (EM) performance. The ceramic thickness at the base of the coil holder is minimized to reduce its purely series thermal impedance, but it must be thick enough to maintain structural integrity and prevent adverse consequences on EM performance from an increased airgap. The optimal fin thickness between conductors is determined through steady-state thermal finite element analysis (FEA) studies to ensure efficient heat dissipation.

#### Interleaved Copper Sheets

Another key feature of the invention is the use of interleaved copper sheets within the iron lamination stack. These thin copper sheets provide a thermal shunt from the coils to the heatsink, significantly reducing the thermal impedance. The copper sheets are periodically inserted within the lamination stack, and the optimal copper fill percentage is determined to balance thermal performance and electromagnetic performance.

A 3D FEA study is conducted to evaluate the impact of copper fill on the motor's thermal and electromagnetic performance. The results show that a 20% copper fill in the lamination stack reduces the equivalent thermal resistance from the conductor to the heatsink by 40% compared to a baseline motor with no copper fill. This improvement in thermal performance is achieved without a significant reduction in torque production, making it an ideal solution for eVTOL applications.

#### Additively Manufactured Heatsink

An additively manufactured heatsink is used to further enhance the thermal performance of the motor. This heatsink is designed to improve air-side heat transfer over the baseline radial fin heat sink through increased surface area. The additively manufactured design allows for the creation of complex geometries that maximize heat dissipation, ensuring that the motor operates at optimal temperatures even under high-power conditions.

### Electromagnetic Performance

#### Flux Density Distribution

A 3D FEA analysis is performed to study the flux density distribution in the motor with the integrated magnetics, insulation, and cooling architecture (MAGICA). The analysis shows that the axially varying interleaved copper and iron yoke laminations necessitate a 3D analysis due to non-zero axial flux components. The peak flux density in the yoke is evaluated for different copper lamination stacking factors. At 0% copper stack factor (no copper), the peak flux density is 1.05-1.1 T. When the copper stack factor is increased to 20%, the peak flux density increases to 1.3 T, resulting in a slight increase in iron losses.

#### Impact on Torque Production

The impact of the copper lamination stack factor on torque production is evaluated using the 3D FEA model. The results show that the torque produced at 0% copper stack factor is 790 Nm, while at 100% copper stack factor, the torque drops to 473 Nm. However, maintaining the copper stack factor below 20% results in a minimal impact on torque production, with a reduction of only 1%. This indicates that the integrated architecture can significantly improve thermal performance without compromising the motor's torque capabilities.

#### Iron Losses

Iron losses in the motor are evaluated using a 2D FEA simulation and the Steinmetz model. The machine designed without MAGICA has 6,893 W of copper losses and 667 W of iron losses at the rated operating condition. The core air-gap flux density is evaluated to be 1.06 T. When integrated with MAGICA at a 20% copper stacking factor, the peak flux density increases to 1.3 T, resulting in an increase in iron losses to 1,095 W. Despite this increase, the overall thermal performance improvement outweighs the additional iron losses, making the integrated architecture a viable solution for eVTOL applications.

#### Eddy Current Losses

Eddy current losses in the yoke are evaluated using a 3D FEA model. The air-gap side edges of the iron laminations experience saturation due to fringing effects at the copper and iron interface, leading to higher eddy current losses in the yoke. The peak eddy current density for a 16.6% copper lamination fill factor is 0.7 Arms/mm², and for a 33.3% copper stacking factor, it is 2.3 Arms/mm². These values are much smaller than the traditional current density limits, indicating that the eddy current losses are negligible compared to the iron losses. Therefore, a 20% copper lamination stack factor is chosen for prototype development as it does not generate significant eddy current loss or a significant reduction in torque production.

### Prototype Development and Preliminary Experimental Results

A prototype of the ceramic coil holder and the MAGICA yoke is developed to demonstrate the thermal performance improvements. The ceramic coil holder is manufactured using waterjet cutting from aluminum nitride (AlN) plates. The thickest available AlN material (7 mm) is sourced to minimize the number of machining operations. Test arrays with varying fin thicknesses are cut to calibrate the waterjet and determine the minimum achievable dimensions. A minimum successful fin thickness of 0.015" is achieved, but final fin and base thicknesses of 0.030" are chosen to provide structural margin and optimize thermal performance.

A 4.5" long, 42.5° pole-pair prototype of the MAGICA yoke is built using 0.010" thick HF-10 C5 insulated iron laminations and 0.005" thick Copper 110 alloy foil. The copper laminations are folded, compressed, and cut to the correct geometry via wire EDM. One copper lamination is periodically placed for every two HF-10 laminations to achieve a 20% copper fill in the lamination stack. Both sets of laminations are coated with EB-548 bonding epoxy before curing.

A control group of the baseline, pure iron yoke is also built for side-by-side thermal performance comparisons. A simple experiment is conducted to qualitatively compare the thermal performance of the two yokes. The test setup consists of the two samples, a ceramic heater, and a thermal camera to monitor the outer surface temperatures. The heat load is applied to the inner surface of each test article, and the results show faster heat spread, more uniform temperature distribution, and lower hotspot temperatures in the MAGICA stack.

### Conclusions and Future Work

The present invention proposes a novel integrated magnetics, insulation, and cooling architecture (MAGICA) for improving the thermal and insulation performance of electric motors, particularly for use in electric vertical takeoff and landing (eVTOL) aircraft. The invention includes a 300 kW slotless PM motor designed with and without MAGICA to illustrate the performance benefits gained with the integrated architecture.

Analysis shows that an optimal copper stacking factor of 20% can reduce the thermal impedance from windings to heatsink by 50% with a maximum torque reduction of 1%. The ceramic holder also provides additional insulation performance improvements, with the potential to limit partial discharge (PD) activity. Future work will focus on further demonstrating the benefits of MAGICA with hardware development, integrating an additively manufactured heatsink, and quantifying power density improvements and PD reduction.

The invention represents a significant advancement in the field of electric motors for eVTOL applications, offering improved thermal management and electrical insulation, thereby enhancing the reliability and performance of these critical systems.