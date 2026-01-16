Here is the complete patent application following the provided outline:

## TECHNICAL FIELD  
The present invention relates generally to smart lighting systems, and more particularly to systems and methods for occupancy detection and lighting control using color-controllable light-emitting diode (LED) fixtures and color sensors. The invention enables estimation of spatial occupancy distribution within an indoor environment while preserving privacy through non-imaging sensing techniques. The disclosed systems and methods utilize modulated light perturbations and light transport analysis to determine room occupancy for optimized lighting control.

## BACKGROUND  
Traditional lighting systems have evolved from incandescent bulbs to more energy-efficient fluorescent and LED fixtures. A current challenge in lighting technology involves developing intelligent systems capable of dynamically adjusting illumination based on real-time occupancy information. While imaging sensors like cameras can provide detailed occupancy data, they raise significant privacy concerns. Non-imaging alternatives such as passive infrared (PIR) sensors have limitations in accuracy and functionality.

Modern LED lighting systems present new opportunities through their color controllability and rapid modulation capabilities. These characteristics enable visible light communication (VLC) techniques where lighting fixtures can simultaneously provide illumination and transmit data. However, existing systems fail to effectively leverage these capabilities for privacy-preserving occupancy detection and lighting optimization.

Current approaches suffer from several limitations: (1) imaging-based systems compromise privacy, (2) non-imaging systems provide insufficient spatial information, (3) existing solutions cannot effectively distinguish between different occupancy scenarios, and (4) current methods lack the ability to model light transport for accurate occupancy estimation. There exists a need for a lighting control system that can accurately detect spatial occupancy distribution while maintaining privacy and enabling optimized illumination.

## SUMMARY  
The present invention provides a smart lighting system comprising color-controllable LED fixtures and non-imaging color sensors configured to estimate spatial occupancy distribution through analysis of modulated light transport. The system operates through alternating sensing and adjustment stages. During sensing, imperceptible perturbations are introduced to base lighting conditions, and sensor responses are measured to construct a light transport matrix. This matrix is analyzed using either a light blockage model (for wall-mounted sensors) or light reflection model (for ceiling-mounted sensors) to estimate occupancy distribution.

Key innovations include: (1) a perturbation modulation technique that enables light transport measurement without noticeable lighting changes, (2) a computationally efficient method for constructing and analyzing the light transport matrix, (3) two distinct occupancy estimation approaches tailored for different sensor configurations, and (4) a complete system architecture that integrates sensing, processing, and lighting control.

The system provides significant advantages over prior solutions: (1) preserves privacy by using non-imaging sensors, (2) enables spatial occupancy estimation unavailable with conventional sensors, (3) operates with existing lighting infrastructure, and (4) supports real-time implementation with modern LED response characteristics. Applications include energy-efficient lighting control for offices, homes, warehouses, and other indoor spaces where occupancy-aware illumination is desired.

## DETAILED DESCRIPTION  

### Rank Minimization  
The light transport matrix A represents the relationship between fixture inputs and sensor outputs, capturing how light propagates through the environment. This matrix typically exhibits low-rank structure due to physical constraints in light transport. The present invention employs rank minimization techniques to efficiently estimate A from limited measurements.

The system acquires data by applying n perturbation patterns δx₁, δx₂, ..., δxₙ to base lighting x₀ and measuring corresponding sensor responses δy₁, δy₂, ..., δyₙ. These measurements form matrices X = [δx₁, δx₂, ..., δxₙ] and Y = [δy₁, δy₂, ..., δyₙ] related by Y = AX. When n exceeds the input dimension m₁, the system computes the Moore-Penrose pseudoinverse: A = YXᵀ(XXᵀ)⁻¹. For underdetermined cases (n < m₁), the invention employs recursive least squares or sparse approximation methods to exploit matrix low-rank properties.

The rank minimization approach enables accurate light transport estimation with minimal measurements, crucial for real-time operation. The system compensates for ambient light by differential measurements between base and perturbed states, eliminating the need for absolute calibration. This method proves particularly effective given the typically small dimensions of fixture-sensor systems compared to projector-camera setups.

### Perturbation-Modulated Lighting  
The invention introduces a novel perturbation modulation technique that enables light transport measurement without perceptible lighting changes. Perturbation patterns δx are carefully designed to meet three requirements: (1) sufficient variation for information capture, (2) small magnitude to avoid human detection, and (3) adequate strength for reliable sensor measurement.

The system employs random perturbation patterns with magnitude ρ = maxᵢ||δxᵢ||∞ optimized through sensitivity analysis. Experimental results indicate ρ = 0.025 (relative to [0,1] input range) provides optimal balance between imperceptibility and measurability. To further minimize perceptibility, the invention implements a traveling salesman problem (TSP) optimization to order perturbations such that consecutive patterns exhibit minimal variation.

This perturbation scheme operates within a two-stage control framework: (1) sensing stage applies modulated perturbations to estimate occupancy, and (2) adjustment stage updates base lighting based on occupancy estimates. The quasi-static assumption ensures lighting changes occur slowly relative to rapid perturbation measurements, maintaining occupant comfort while enabling continuous monitoring.

### Analysis of the Light Transport Matrix  
The system analyzes changes in the light transport matrix to estimate occupancy. For an empty reference room with transport matrix A₀, current matrix A yields difference matrix E = A₀ - A. The invention provides two analysis methods depending on sensor placement:

For wall-mounted sensors (direct path available), entries in E indicate blocked direct paths between fixtures and sensors. The system aggregates E to an Nₛ × Nₗ matrix Ê through weighted channel summation. Occupancy confidence at 3D point P is computed as:

C(P) = ΣᵢΣⱼ ÊᵢⱼG(dᵢⱼ(P),σ)/ΣᵢΣⱼ G(dᵢⱼ(P),σ)

where G is a Gaussian kernel, dᵢⱼ(P) is distance from P to fixture j-sensor i path, and σ controls smoothing. This formulation relates to inverse Radon transform, enabling rough 3D reconstruction from sparse measurements.

For ceiling-mounted sensors (reflection paths only), the system precomputes reflection kernels Rᵢⱼ describing how floor point reflectivity affects each fixture-sensor pair. The 2D confidence map is computed as:

M = ΣᵢΣⱼ Êᵢⱼ^λ¹ Rᵢⱼ/ΣᵢΣⱼ Rᵢⱼ^λ²

where λ₁ ≥ 1 sharpens large entries and λ₂ ≥ 0 normalizes non-uniform sensor coverage.

### Volume Rendering  
The 3D reconstruction approach generates occupancy volumes by evaluating confidence C(P) across discretized space. The system employs several optimizations for practical implementation:

1. Precomputation of Gaussian kernels for all fixture-sensor pairs
2. Hashing-based acceleration using spatial partitioning
3. Parallel computation on GPU hardware
4. Selective refinement in high-interest regions

For 2D floor-plane estimation, the reflection model computes confidence maps at 1-inch resolution. Both methods produce sufficiently accurate results for lighting control while maintaining computational efficiency. The system adapts reconstruction parameters (σ, λ₁, λ₂) based on desired precision and available processing resources.

Experimental results demonstrate effective occupancy estimation across various scenarios. The 3D approach achieves superior performance (mean correlation 0.45 vs. 0.30 for 2D), particularly when distinguishing multiple occupied regions. Both methods enable lighting control decisions while preserving privacy through intentionally limited reconstruction precision.