# DESCRIPTION

## TECHNOLOGICAL FIELD

The present invention relates to the field of resistive crossbar memory systems, particularly to methods and apparatus for performing efficient and power-effective readout operations in such systems. More specifically, the invention addresses the challenges posed by sneak-paths, which are parasitic currents that can significantly degrade the performance and reliability of crossbar-based memory arrays. The invention provides a comprehensive solution that leverages the spatial correlation of sneak-paths and employs adaptive-threshold readout techniques to mitigate these issues.

## BACKGROUND

Resistive crossbar memory systems are widely recognized for their potential in high-density, non-volatile memory applications. These systems consist of a grid of intersecting wires, with resistive memory elements (such as memristors) located at the intersections. The primary advantage of such systems is their ability to store large amounts of data in a compact form, making them ideal for applications requiring high storage density and low power consumption.

However, resistive crossbar memory systems face significant challenges, particularly related to sneak-paths. Sneak-paths are parasitic currents that flow through unintended paths in the crossbar array, leading to increased power consumption and data corruption. These currents are data-dependent and can vary significantly across different cells, making it difficult to predict and control their impact. As a result, direct memory readout is often unreliable, necessitating the development of advanced readout techniques to ensure accurate and efficient data retrieval.

One of the key properties of sneak-paths is their spatial correlation. The value of a sneak-path at one location in the crossbar can be used to estimate the values at other correlated locations. This property can be exploited to develop faster and more power-efficient readout techniques. Additionally, the memory-locality property of computer systems, where data is typically accessed in blocks of contiguous bits, can be leveraged to enhance the effectiveness of these readout techniques.

Despite the advances in the field, there remains a need for a robust and efficient method to perform readout operations in resistive crossbar memory systems while minimizing the impact of sneak-paths. The present invention addresses this need by providing a novel approach that combines the spatial correlation of sneak-paths with adaptive-threshold readout techniques to achieve high-performance and low-power readout operations.

## BRIEF SUMMARY

The present invention provides a method and apparatus for efficiently performing readout operations in a resistive crossbar memory system. The invention leverages the spatial correlation of sneak-paths and employs adaptive-threshold readout techniques to mitigate the impact of parasitic currents and ensure reliable data retrieval.

In one aspect, the invention involves a method for performing readout operations in a resistive crossbar memory system. The method includes analyzing the spatial correlation of sneak-paths in the crossbar array, determining an adaptive threshold for readout based on the sneak-paths correlation, and performing readout operations using the adaptive threshold. The method further includes techniques for multi-read of initial bits, the use of predefined dummy bits, and optimization of power consumption.

In another aspect, the invention provides an apparatus for performing readout operations in a resistive crossbar memory system. The apparatus includes a crossbar array, a readout circuit configured to analyze the spatial correlation of sneak-paths, and a processing unit configured to determine an adaptive threshold for readout based on the sneak-paths correlation. The apparatus is further configured to perform readout operations using the adaptive threshold and to optimize power consumption.

The invention also includes a detailed description of the operations performed by a computing device to efficiently perform readout operations in a resistive crossbar memory system, including the steps involved in analyzing sneak-paths, determining adaptive thresholds, and performing readout operations.

## DETAILED DESCRIPTION

### Sneak Paths Analysis

Sneak-paths are parasitic currents that flow through unintended paths in a resistive crossbar memory array. These currents can significantly impact the performance of the system by increasing power consumption and causing data corruption. The impact of sneak-paths is twofold: first, a considerable amount of undesirable energy is consumed as current sneaks throughout the array cells; second, the sneak currents are data-dependent and cannot be predicted accurately. This leads to distributions that represent the "One" and "Zero" values rather than single values, as shown in Fig. 1c. The magnitude of the sneak-current is typically higher than the current of the desired memory cell, resulting in highly overlapped distributions for the two binary values. Direct memory readout is therefore not possible, necessitating a power-efficient sneak-paths immune readout technique.

One of the generally utilized properties of sneak-paths current is its spatial correlation. Knowing the sneak-path noise value at one location of the crossbar helps to estimate the values at other correlated locations. This property can be effectively utilized to develop faster and more power-efficient readout techniques for resistive crossbar memories. In general, a crossbar can be accessed using two modes: "floating terminals" and "connected terminals." In the "floating terminals" approach, the selected array terminals are kept floating, while in the "connected terminals" approach, the selected rows and columns are connected to two common nodes. The two extra nodes can be used as access terminals to the array or to enforce a bias voltage. This allows for better control of the sneak-paths behavior and yields a more usable equivalent circuit. In such a case, the sneak-paths are represented by three lambed resistances ('Rr', 'Ra', and 'Rc') as shown in Fig. 2d.

Understanding the correlation of these elements over the crossbar facilitates better handling of the sneak-paths noise. For instance, 'Rr' is a parallel combination of all the desired row cells apart from the desired one and is given by the equation:

\[ Rr = \frac{1}{\sum_{x=1}^{L-1} \frac{1}{Rx}} \]

where 'Rx' is the resistance of a one-row cell, and 'L' is the array length. The row cell resistance can be either 'Ron' or 'Roff', which are the ON and OFF resistance of the device under 'Vn1 - Vn4' voltage drop, respectively. The row resistance can be rewritten as:

\[ Rr = \frac{1}{\frac{Non}{Ron} + \frac{L - Non - 1}{Roff}} \]

where 'Non' is the number of ON cells within the accessed row not counting the accessed cell itself. The remaining two sneak-path components (Rc and Ra) have similar expressions. In the case of biasing the unused array terminals, the sneak-path component 'Ra' is shorted out. It should be noted that although the metal line resistances are not included in the equivalent circuit for simplicity, they have been fully considered in the simulations carried out in this work.

For practical array sizes, the values of 'Rr' and 'Rc' are almost constant over the same row or column, respectively. For instance, the sneak-paths row resistances found at two different locations in the same row have all cells in common except the two cells that are swapped because of the accessed locations. For devices with a large OFF/ON ratio, the relative change in the sneak-paths row resistance is given by:

\[ \Delta R / R = \frac{2 \cdot Ron \cdot Roff}{(Ron + Roff)^2} \]

where 'ρ' is the OFF/ON ratio of the used device. The maximum relative change in the row resistance versus the array size for a balanced number of zeros and ones is plotted in Fig. 3a. The figure shows that as the array size increases, the effect of a single bit swap diminishes. The other parameter that affects ΔR/R is the number of ones (per row or column), as given by the equation. Figure 3b shows that the maximum relative change of sneak-paths resistance is still small while the percentage of ones per row/column is swept. Hence, 'Rr' is almost constant over a given row, and 'Rc' is almost constant over a given column. Given the randomness of the data, 'Rr' and 'Rc' are considered two independent random variables.

### Sneak-Paths Correlation

The spatial correlation of sneak-paths is a critical property that can be exploited to improve the efficiency and accuracy of readout operations in resistive crossbar memory systems. By understanding the correlation of sneak-paths over the crossbar, it is possible to develop more effective readout techniques that minimize the impact of parasitic currents.

In the "connected terminals" crossbar, the values of 'Rr' and 'Rc' can be safely shared over the same row or column, respectively. This is equivalent to defining an adaptive threshold that changes at each new row readout, which can be achieved with the aid of the "connected terminals" crossbar. The generic "connected terminals" circuit model shown in Fig. 2d can be simplified for the case of 'VB' terminals bias. Terminals 'n3' and 'n4' are connected to 'VB', and terminals 'n1' and 'n2' are connected to 'VDD' and virtual ground. This can be done with two different implementations as shown in Fig. 4. Using a virtual ground sensing circuit forces all of the array elements to have a defined voltage drop independent of the data stored in the array. The desired cell experiences a full 'VDD' voltage drop, while the sneak-paths components of 'Rr' and 'Rc' have a voltage drop of 'VDD - VB'. Because of the device saturation nonlinearity, the full voltage drop on the desired cell makes the magnitude difference between its ON and OFF states much larger than any error introduced by sharing 'Rr' or 'Rc' over a segment. While both of 'Rr' and 'Rc' drain parasitic sneak-current, the current leak through only one of them affects the correctness of the readout operation. When the read circuit is connected to node 'n1', as shown in Fig. 4b, the sense current (Isense) is defined as:

\[ Isense = Im - Ir \]

where 'Im' is the desired current and 'Ir' is the row sneak current component. Sensing from node 'n2' swaps the locations and the role of 'Rr' and 'Rc' in the circuit, as shown in Fig. 5a. The sense current is shifted from its desired value by the sneak-current of the row or the column. However, this shift is constant within a given row or column, based on the connection orientation.

### Adaptive-Threshold Readout

The spatial correlation of sneak-paths can be effectively utilized in the case of sequential reading for the stored data on a memory array. The good news is that this is the typical memory access scheme in computer systems. Because of the memory-locality property, data is transferred and shared between different memory layers as a block of contiguous bits, rather than in random bits or words. This locality property is of help only if the knowledge gained from reading a single bit can be adopted in reading its neighborhoods. This is true for the "connected terminals" crossbar, where the values of 'Rr' and 'Rc' can be safely shared over the same row or column, respectively, as discussed in the previous sections. This is equivalent to defining an adaptive threshold that changes at each new row readout, which can be achieved with the aid of the "connected terminals" crossbar.

The generic "connected terminals" circuit model shown in Fig. 2d can be simplified for the case of 'VB' terminals bias. Terminals 'n3' and 'n4' are connected to 'VB', and terminals 'n1' and 'n2' are connected to 'VDD' and virtual ground. This can be done with two different implementations as shown in Fig. 4. Using a virtual ground sensing circuit forces all of the array elements to have a defined voltage drop independent of the data stored in the array. The desired cell experiences a full 'VDD' voltage drop, while the sneak-paths components of 'Rr' and 'Rc' have a voltage drop of 'VDD - VB'. Because of the device saturation nonlinearity, the full voltage drop on the desired cell makes the magnitude difference between its ON and OFF states much larger than any error introduced by sharing 'Rr' or 'Rc' over a segment. While both of 'Rr' and 'Rc' drain parasitic sneak-current, the current leak through only one of them affects the correctness of the readout operation. When the read circuit is connected to node 'n1', as shown in Fig. 4b, the sense current (Isense) is defined as:

\[ Isense = Im - Ir \]

where 'Im' is the desired current and 'Ir' is the row sneak current component. Sensing from node 'n2' swaps the locations and the role of 'Rr' and 'Rc' in the circuit, as shown in Fig. 5a. The sense current is shifted from its desired value by the sneak-current of the row or the column. However, this shift is constant within a given row or column, based on the connection orientation.

### Multi-Read for Initial Bits

Each bit generally has two unknowns: 'Rm' and 'Rr' (or 'Rc'). Without adopting sneak-paths correlation and locality, multiple access stages are needed to estimate the bit value. However, a faster readout can be achieved by categorizing the bits into two types: the "initial bits," which are the first bits accessed in a given column, and "regular bits," which are any other bits in the array. To estimate the value of the "initial bit," two unknowns need to be solved, namely the desired resistance (Rm) and the row sneak resistance (Rr). However, the remaining bits in the row share the same 'Rr' value, and 'Ir' is treated as the significant sneak-path component for a given row. Any of the readout techniques presented in the literature can be used to estimate the "initial bit." These "initial bits" readout dictates the threshold used for the remaining bits in that row. Figure 5a shows the readout sequence for the array when "initial bits" strategy is adopted. Therefore, the first (initial bit) could be any bit in the array that requires 'n' stages of reading. The rest of the bits in the same row are then accessed in sequence, only one time for each. Reading from the next row requires a new "initial bit," which in this case is the first bit in the row, as shown in Fig. 5a. The same sequence is followed until the fetched data block for the cache is completed, i.e., each row contains one "initial bit," and the rest of the bits are accessed in a single stage fashion. For a contiguous block of data readout using the "initial bits" technique, the proposed readout procedure is given as follows:

**Case 1: The first accessed bit in the row 'i' (the initial bits):**
Use a multi-stage readout technique to estimate the desired cell current \( I_{m} \) and the row sneak-current component \( I_{r} \).

**Case 2: Accessing the rest of the bits in the same row:**
Access the desired cell for a single time to estimate its value, where \( I_{sense} = I_{m} - I_{r} \).

where 'i' and 'j' are the desired row and column, respectively. It should be noted that in the case of sensing from 'n1' data is accessed in a column-wise rather than row-wise scheme.

The readout circuitry for the "initial bit" is made of two parts: a virtual-ground ADC for the current sensing, and a digital processing circuitry for calculating the "initial bit" parameters and doing the threshold comparisons. Typically, a single readout circuitry is needed per memory array. This does not impact the whole memory density as presented in previous works.

### Predefined Dummy Bits

A more time-efficient way to estimate the adaptive threshold is to add "dummy bits" with predefined values to the array. The general concept of adding predefined bits to an array for sneak-paths estimation is presented in the literature. In our case, for a "dummy bit," the value of 'Rm' is known in advance, and a single readout is needed to estimate the value of 'Rr'. This estimated 'Rr' value is reused with the other bits in the same row. A single readout is required in this case to estimate the remaining unknown (Rm). This value is used for the rest of the bits in the same row. The "dummy bit" can be organized in several ways, given that each row contains a single bit. Figure 5b shows a possible organization of dummy bits that is suitable for a row-wise readout analogy. For a contiguous block of data readout using the "dummy bits" technique, the proposed readout procedure is given as follows:

**Case 1: Accessing the "dummy bit" of row 'i':**
Estimate the sneak-path row component using a single array access, where \( I_{sense} = I_{dummy} + I_{r} \).

The current \( I_{dummy} \) can be used without any processing, since the values of \( I_{dummy} \) is a DC shift that can be compensated in the comparison process.

**Case 2: Accessing the rest of the bits in the same row:**
Access the desired cell for a single time to estimate its value, where \( I_{sense} = I_{m} - I_{r} \).

where 'i' and 'j' are the desired row and column, respectively. The dummy current \( I_{dummy} \) is known in the design time, where it can be \( I_{on} \) or \( I_{off} \) depending on which value is used to be stored in the dummy cells. Moreover, a dummy cell could be just a reference static resistor rather than a memristor, since there is no need to write it after the array fabrication.

The "dummy bits" technique adds a small overhead to the readout process, as a "dummy bit" needs to be accessed a single time (in comparison to 'n' times for an "initial bit"). However, for practical size arrays of 256 k size or more, the average number of array accesses per bit that occurs when fetching a block of data from memory is almost one for both methods. Figure 6a shows the average number of readouts per memory bit, where the overhead is shared over "regular bits," versus the fetched data size. It also illustrates how the average number of readouts converges to one very fast. The ripples in the curve occur because that start reading from a new row adds extra overhead of an "initial bit" or a "dummy bit." It should be noted that the typical cache line is 0.5 kb (64 bytes), where multiple lines are fetched from memory in sequence based on the cache policy. This value is much larger in the case of RAM fetching from HDD. While the "dummy bits" technique exhibits a better behavior, it comes at a small cost to the effective area of the array, as "dummy bits" are not used to store real data. This negligible overhead is shown in Fig. 6b.

The readout circuitry for the "dummy bits" technique can be implemented in two ways. The first approach is to use an analog circuit for current sensing and a simple digital circuit for comparisons and estimation, as discussed in the "initial bits" readout. Typically, most of the readout circuit area in this methodology is consumed by the conversion of the data from one domain to the other using ADCs. A more area-efficient implementation is to adopt a totally analog compensated readout circuit, as presented in previous work. In this approach, the current of a "dummy cell" is sampled on a first capacitor, and the sensed current from each desired cell is sampled on a second one in sequence. Comparison between the two capacitor voltages leads to estimating the stored data in the desired memory cells.

### Crossbar Power Consumption

Undesirable sneak-paths power consumption is not avoidable in high-density gateless arrays. However, it can be reduced by utilizing devices with nonlinear saturation behavior. Figure 8 shows the 'i-v' hysteresis of two of our fabricated devices. The second device shows higher saturation nonlinearity than the first one. Reducing the voltage applied to such devices by fifty percent can increase its saturation resistance up to two orders of magnitude. This is a very attractive property since a sneak path is made of a series of memristor devices, where a sub-voltage is dropped on each of them. In the "connected terminals" structure, the device nonlinearity can be enforced by biasing the unused terminals to sub-read voltage. In such a case, the very small 'Ra' is shorted out, and the nonlinearity of the other terminals efficiently utilized. Figure 9a shows that the optimal selection is made by biasing the unused terminals voltage to be \( VB = VDD/2 \). The power consumption of this method is almost the same as the baseline "floating terminals," as shown in Fig. 9b. The figure also shows the great power-saving of the "connected terminals" while comparing it with the power-hungry "grounded terminals" technique. It should be noted that power consumption saturates for larger array sizes because of the crossbar metal lines.

### Figure-of-Merit

In general, the presented technique offers a sneak-paths immune readout that is more power-efficient and faster than the state-of-the-art crossbar accessing techniques presented in the literature. Table 1 shows a detailed comparison of the various gateless techniques that can provide an error-free readout. The different methods are compared based on a figure-of-merit (FoM), which is defined as:

\[ \text{FoM} = \frac{\text{Speed} \times \text{Power Efficiency}}{\text{Complexity}} \]

where the proposed technique shows the best FoM.

### Operations Performed by a Computing Device to Efficiently Perform Readout Operations

The operations performed by a computing device to efficiently perform readout operations in a resistive crossbar memory system include the following steps:

1. **Analyze Sneak-Paths Correlation:**
   - The computing device analyzes the spatial correlation of sneak-paths in the crossbar array. This involves measuring the sneak-path currents at various locations and using this information to estimate the values at other correlated locations.

2. **Determine Adaptive Threshold:**
   - Based on the sneak-paths correlation, the computing device determines an adaptive threshold for readout. This threshold is adjusted dynamically for each row or column to account for the variations in sneak-path currents.

3. **Perform Multi-Read for Initial Bits:**
   - The computing device performs a multi-stage readout for the "initial bits" in each row. This involves multiple accesses to estimate the desired cell current and the row sneak-current component.

4. **Use Predefined Dummy Bits:**
   - The computing device uses predefined dummy bits with known values to estimate the sneak-path row component. This reduces the number of readout stages required for the remaining bits in the same row.

5. **Optimize Power Consumption:**
   - The computing device optimizes power consumption by utilizing devices with nonlinear saturation behavior and biasing the unused terminals to sub-read voltage. This reduces the impact of sneak-paths and minimizes power consumption.

6. **Perform Readout Operations:**
   - The computing device performs readout operations using the adaptive threshold and the optimized power consumption settings. This ensures accurate and efficient data retrieval from the crossbar array.

By following these steps, the computing device can efficiently perform readout operations in a resistive crossbar memory system, ensuring high performance and low power consumption.