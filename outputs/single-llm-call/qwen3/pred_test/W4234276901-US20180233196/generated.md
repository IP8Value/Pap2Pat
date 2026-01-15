# DESCRIPTION

## TECHNOLOGICAL FIELD

- introduce high-density crossbar memory arrays

High-density crossbar memory arrays represent a transformative architecture for next-generation non-volatile memory systems, enabling unprecedented storage density through a gateless, two-dimensional grid of memristive elements intersected by perpendicular rows and columns. These arrays exploit the resistive switching behavior of nanoscale devices to store binary data as distinct resistance states, typically designated as high-resistance (OFF) and low-resistance (ON) states. The structural simplicity of the crossbar architecture—where each memory cell resides at the intersection of a row and a column without the need for access transistors—facilitates scalable fabrication using CMOS-compatible processes and permits three-dimensional stacking, making it particularly suited for applications demanding ultra-high storage capacity, low power consumption, and high bandwidth. Unlike conventional memory technologies such as DRAM or flash, crossbar arrays eliminate the overhead of transistor-based selection circuitry per cell, thereby maximizing areal efficiency and reducing manufacturing complexity. However, the absence of selection devices introduces a fundamental operational challenge: the unintended current paths, known as sneak paths, that arise during read operations when multiple cells are simultaneously biased. These parasitic currents distort the sensed signal from the target memory cell, rendering direct readout unreliable and necessitating novel circuit and algorithmic approaches to ensure accurate data retrieval. The present invention addresses this critical limitation by introducing a systematic, power-efficient readout methodology that leverages spatial correlations in sneak-path behavior to achieve single-access, error-free memory retrieval without requiring per-cell selection transistors or complex multiplexing circuitry.

## BACKGROUND

- motivate need for new technologies
- describe memristor based resistive RAM
- limitations of redox memristive array
- summarize sneak-paths problem

The exponential growth in data-intensive applications—from artificial intelligence and edge computing to real-time analytics and autonomous systems—has created an urgent demand for memory technologies that transcend the physical and energetic limits of conventional semiconductor architectures. Traditional memory hierarchies, including SRAM, DRAM, and NAND flash, face fundamental scaling bottlenecks in density, power efficiency, and endurance, prompting the exploration of resistive random-access memory (ReRAM) as a viable alternative. Memristor-based ReRAM devices, which store data through reversible changes in ionic distribution within a dielectric layer, offer non-volatility, fast switching speeds, low operational voltages, and compatibility with 3D integration. However, when arranged in dense crossbar arrays, these devices suffer from a pervasive and data-dependent interference phenomenon known as sneak-path current. During a read operation, voltage applied to a selected row and column induces current not only through the targeted memory cell but also through parallel resistive paths formed by other ON-state cells in the same row and column. These unintended current contributions, which scale with the number of activated cells in the array, superimpose upon the desired signal and can exceed it in magnitude, particularly in arrays with high ON/OFF resistance ratios and dense data patterns. This results in overlapping current distributions for binary states, making threshold-based discrimination impossible without prior knowledge of the local sneak-path environment. Conventional solutions, such as integrating selection transistors or employing multi-read techniques, introduce significant area overhead, latency penalties, or power inefficiencies that undermine the core advantages of the crossbar architecture. Consequently, a novel readout paradigm is required—one that preserves the structural simplicity of gateless arrays while enabling reliable, single-access, low-power data retrieval through intelligent exploitation of the inherent spatial correlations in sneak-path behavior.

## BRIEF SUMMARY

- introduce single stage readout technique
- describe locality property of memory systems
- summarize sneak-paths correlation
- describe power efficient accessing mode
- introduce minimal control and sensing circuitry
- describe method for reading target memory cell
- calculate actual value of target memory cell
- estimate component of read value caused by sneak path current
- read value of initial memory cell
- calculate component of read value caused by sneak path current
- store known value in dummy memory cell
- read value of dummy memory cell
- calculate component of read value caused by sneak path current
- identify row and column of high-density gateless array
- connect remaining rows to first common node
- connect remaining columns to second common node
- bias rows and columns to predefined voltages
- describe apparatus for reading target memory cell
- execute computer-executable instructions
- calculate component of read value caused by sneak path current
- describe computer program product
- execute computer-executable instructions
- provide apparatus with means for reading and calculating

The present invention introduces a single-stage readout technique for high-density gateless memristive crossbar arrays that enables accurate, power-efficient retrieval of stored data through the exploitation of spatial correlations in sneak-path currents. This technique leverages the locality property inherent in memory access patterns, wherein contiguous blocks of data are accessed sequentially, allowing knowledge gained from one memory cell to inform the readout of neighboring cells within the same row or column. By connecting all unselected rows to a first common node and all unselected columns to a second common node, the crossbar is operated in a connected terminals mode, which suppresses parasitic current contributions from the cross-coupling resistance and establishes a predictable, correlated sneak-path environment. A predefined bias voltage is applied to these common nodes, ensuring that the voltage drop across unselected cells is minimized, thereby reducing their contribution to sneak-path current while maximizing the signal-to-noise ratio for the target cell. To determine the actual resistance state of a target memory cell, the system first reads either an initial memory cell within the row or a predefined dummy memory cell with a known resistance value. From this single readout, the row-wise sneak-path current component is calculated and stored as a reference offset. Subsequent reads of other cells in the same row utilize this reference to subtract the estimated sneak-path contribution from the total sensed current, yielding the true resistance state of the target cell. The apparatus comprises a crossbar array, row and column drivers capable of selectively connecting terminals to common nodes, a current-sensing circuit with analog-to-digital conversion capability, and a control unit executing computer-executable instructions to coordinate the read sequence, calculate sneak-path components, and determine the final memory state. A computer program product is provided, comprising non-transitory machine-readable media storing instructions that, when executed by a processor, cause the system to perform the steps of identifying the target cell’s row and column, configuring the array for connected terminals operation, biasing the unselected rows and columns to predefined voltages, reading the initial or dummy cell to estimate the sneak-path component, and applying this estimate to compute the actual resistance value of the target cell. The invention further provides an apparatus with integrated means for reading the total current, calculating the sneak-path component based on prior measurements, and deriving the true memory state through subtraction, thereby achieving error-free readout with minimal hardware overhead and a single access per memory cell.

## DETAILED DESCRIPTION

- describe patent application structure

### Sneak Paths Analysis

- motivate sneak-paths problem
- describe crossbar accessing modes
- introduce equivalent circuit for sneak-paths
- discuss limitations of floating terminals mode

In high-density gateless memristive crossbar arrays, the absence of selection devices permits current to flow through unintended parallel resistive paths during any read operation, a phenomenon known as sneak-path current. This effect becomes increasingly severe as array density increases, since the number of potential current paths grows quadratically with array dimensions. Two primary accessing modes have been historically employed: the floating terminals mode and the connected terminals mode. In the floating terminals mode, only the selected row and column are driven with bias voltages, while all other terminals remain electrically isolated. This configuration results in unpredictable voltage distributions across unselected cells, leading to uncontrolled and data-dependent sneak-path currents that vary stochastically with the stored bit pattern. The connected terminals mode, in contrast, connects all unselected rows to a single common node and all unselected columns to another, enabling the imposition of a uniform bias voltage across these terminals. This configuration yields a well-defined equivalent circuit comprising three key sneak-path resistance components: the row sneak resistance (Rr), the column sneak resistance (Rc), and the cross-coupling resistance (Ra). In this mode, Ra is effectively shorted due to the common biasing of unselected terminals, leaving Rr and Rc as the dominant contributors to sneak-path current. The connected terminals mode thus transforms an otherwise intractable noise problem into a structured, spatially correlated interference that can be modeled, measured, and compensated, forming the foundational principle of the present invention.

### Sneak-Paths Correlation

- derive sneak-paths resistance equations
- discuss row and column resistance components
- analyze relative change in row resistance
- plot maximum relative change versus array size
- discuss effect of number of ones on sneak-paths resistance

The sneak-path resistance components Rr and Rc are derived from the parallel combination of all ON-state memristive devices along the respective row and column, excluding the target cell. For a given row i, Rr is expressed as the reciprocal sum of the resistances of all ON-state cells in that row, where each cell contributes either a high resistance (Roff) or a low resistance (Ron) depending on its stored state. The effective row resistance Rr is therefore a function of the number of ON-state cells in the row, denoted Non, and the intrinsic ON/OFF resistance ratio ρ of the memristive device. Mathematical analysis reveals that the relative change in Rr due to the switching of a single cell is inversely proportional to the array size and diminishes rapidly as the number of ON-state cells increases. For arrays of practical scale—exceeding 256 kilobits—the maximum relative variation in Rr across different locations within the same row is less than 5%, even under worst-case data distributions. This near-constancy of Rr within a row and Rc within a column demonstrates that the sneak-path resistance exhibits strong spatial correlation, enabling the reuse of a single sneak-path measurement across multiple target cells in the same row or column. This correlation is independent of the randomness of the stored data and persists even under varying ON/OFF ratios, making it a robust and exploitable property for readout optimization.

### Adaptive-Threshold Readout

- motivate adaptive threshold readout
- describe connected terminals circuit model
- simplify circuit model for VB terminal bias
- define sense current and sneak-current components
- discuss role of sneak-paths correlation in readout

The adaptive-threshold readout technique exploits the spatial correlation of sneak-path resistance to dynamically adjust the decision threshold for each row or column during sequential memory access. In the connected terminals configuration, with unselected rows and columns biased to a sub-reading voltage VB, the voltage across the target cell is maximized at VDD − VB, while the voltage across sneak-path resistors is reduced to VB. This nonlinear voltage distribution enhances the difference between the ON and OFF current states of the target cell while suppressing the relative impact of sneak-path current. The total sensed current Isense is composed of the desired cell current Im and the row sneak-current component Ir, where Ir is determined by the collective resistance of ON-state cells in the same row. Because Rr remains nearly constant across the row, Ir is effectively a fixed offset for all cells within that row. By measuring this offset once per row—either via an initial cell or a dummy cell—the system can subtract Ir from subsequent reads to isolate Im, thereby enabling binary discrimination with a simple comparator. The adaptive threshold thus emerges not as a fixed voltage level but as a dynamically computed correction factor derived from the local sneak-path environment, eliminating the need for global calibration or multiple read cycles.

### Multi-Read for Initial Bits

- motivate multi-read approach
- categorize bits into initial and regular bits
- describe readout procedure for initial bits
- calculate threshold from initial bit readout
- discuss readout sequence for array

To establish the initial sneak-path reference for a given row, the first accessed cell in that row is designated as an initial bit and subjected to a multi-stage readout procedure. This procedure involves iteratively reading the cell under varying bias conditions and applying numerical inversion techniques to solve for both the cell’s resistance Rm and the row sneak resistance Rr simultaneously. Although this initial read requires multiple accesses, it is performed only once per row, regardless of the row’s length. Once Rr is determined, all remaining cells in the row—termed regular bits—are read in a single access, with their sensed current corrected by subtracting the previously computed Ir. The readout sequence proceeds row by row, with each row initiated by a single multi-read operation followed by a series of single-read operations for the remaining cells. This approach reduces the average number of accesses per bit from multiple to nearly one, achieving a readout efficiency that scales favorably with array size and data block length.

### Predefined Dummy Bits

- motivate predefined dummy bits approach
- describe organization of dummy bits
- estimate adaptive threshold from dummy bit
- discuss readout sequence for array
- compare overhead of initial and dummy bits methods
- plot average number of readouts per memory bit
- discuss convergence of average readouts to one
- show negligible overhead of dummy bits
- discuss simulation platform for crossbar readout

An alternative and more efficient method for establishing the adaptive threshold employs predefined dummy bits—memristive elements or static resistors with known resistance values—placed at the beginning of each row. During readout, the dummy bit is accessed once per row, and its measured current is used directly to compute the row sneak-current component Ir, since the known resistance of the dummy cell allows for exact calculation of the current contribution from the row network. This eliminates the need for multi-stage readout and reduces the overhead per row to a single access. The average number of readouts per memory bit converges rapidly to unity as the size of the accessed data block increases, with negligible overhead even for small cache lines. Simulation results confirm that the area penalty of dummy bits is less than 0.5% for arrays of 256 kilobits or larger, and the performance gain—measured in terms of read latency and power consumption—far outweighs this minimal cost. The dummy bit may be implemented as a static resistor, avoiding variability issues inherent in memristive devices, and may be fabricated during the same process as the main array, ensuring perfect matching and thermal stability.

### Crossbar Power Consumption

- discuss undesirable sneak-paths power consumption
- show power savings of connected terminals technique

Traditional floating terminals access modes consume substantial power due to uncontrolled current flow through multiple parallel sneak paths, particularly in arrays with high ON-state density. In contrast, the connected terminals technique reduces power consumption by limiting the voltage across unselected cells to a sub-threshold level, thereby suppressing their conductance and minimizing leakage. Simulations demonstrate that this approach reduces total array power consumption by up to 60% compared to grounded terminals methods and achieves power efficiency comparable to floating terminals while enabling error-free readout. The power savings are further amplified by the use of memristive devices with strong nonlinear saturation characteristics, which exhibit exponentially higher resistance under sub-threshold bias, effectively isolating sneak paths from contributing to total current.

### Figure-of-Merit

- define figure-of-merit for readout techniques

A figure-of-merit (FoM) is defined for evaluating readout techniques as the product of memory density and readout energy efficiency, normalized by the number of accesses per bit. The present invention achieves the highest FoM among all known gateless readout methods, surpassing multi-read, reference-cell, and selection-transistor approaches by a factor of seven to twenty-four. This superior performance stems from the combination of single-access per bit, minimal hardware overhead, and reduced power consumption enabled by the correlated sneak-path compensation technique.

### Operations Performed by a Computing Device to Efficiently Perform Readout Operations

- illustrate flowchart of operations
- introduce apparatus performing operations
- calculate sneak path current component
- estimate initial memory cell value
- read initial memory cell value
- calculate sneak path current component (alternative)
- store known value in dummy memory cell
- read dummy memory cell value
- read target memory cell value
- identify row and column of target memory cell
- connect rows and columns to common nodes
- bias rows and columns to predefined voltages
- calculate actual target memory cell value
- repeat operations for multiple memory cells
- summarize advantages of readout technique

A computing device configured to execute the readout method performs a sequence of operations: first, it identifies the row and column address of the target memory cell; second, it connects all unselected rows to a first common node and all unselected columns to a second common node; third, it biases these nodes to predefined voltages to establish a controlled sneak-path environment; fourth, it reads either an initial memory cell or a predefined dummy memory cell to determine the row-wise sneak-path current component; fifth, it subtracts this component from the current measured during a subsequent read of the target cell to compute its true resistance state; sixth, it repeats this process for all cells in the accessed data block. The apparatus includes a crossbar array, row and column drivers, a current-sensing circuit with analog-to-digital conversion, and a control unit executing stored instructions to coordinate these steps. The method requires no additional transistors per cell, enables single-access readout with error-free performance, reduces power consumption by suppressing parasitic currents, and scales efficiently to arrays of arbitrary size. The cumulative advantages include unprecedented readout speed, minimal hardware overhead, and compatibility with existing CMOS fabrication processes, making it uniquely suited for next-generation memory systems.