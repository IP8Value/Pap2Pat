Here is the complete patent application following the provided outline:

# DESCRIPTION

## BACKGROUND  

The present invention relates generally to image-based visual hull (IBVH) techniques for three-dimensional object reconstruction. Conventional IBVH methods utilize shape-from-silhouette approaches to generate approximate convex hulls containing target objects from multiple reference images. While these methods offer advantages over traditional geometry-based approaches in terms of data acquisition costs and rendering realism, they suffer from significant limitations in reconstructing concave object surfaces and object bottoms.  

The fundamental limitation of conventional IBVH stems from its reliance on silhouette information alone, which inherently cannot capture concave surface features. Additionally, the requirement that all reference images be registered to the same reference frame makes bottom reconstruction problematic, as bottom images typically cannot be acquired simultaneously with other reference views. Attempts to address these limitations through mirror-based systems have been proposed, but these approaches require fixed relative positions between objects and mirrors and still fail to adequately reconstruct concave surfaces.  

## SUMMARY  

The present invention provides an improved method and apparatus for concave object reconstruction in image-based visual hull systems. The disclosed method utilizes a novel image acquisition platform comprising a planar glass surface and planar mirror to simultaneously capture both top and bottom views of an object within a single reference frame. This platform enables the generation of virtual inside-out images that facilitate accurate reconstruction of concave surfaces and object bottoms without requiring complex alignment calculations.  

An apparatus embodiment includes a storage medium containing reference image data and several processing modules. A pre-processing module segments object silhouettes from background in reference images. A virtual-image synthesis module generates virtual camera parameters and corresponding virtual images for concave surface reconstruction. A point relocation module adjusts three-dimensional coordinates during rendering to properly reconstruct concave regions.  

An alternative apparatus embodiment incorporates these modules within a computing device having specialized memory architecture. The device includes a processor with multiple cores and cache levels, a memory controller, and a memory bus connecting to system memory containing an operating system, applications, and program data including the concave object modeling processing application.  

The method involves generating virtual inside-out images from concave regions using silhouette processing. These virtual images create negative silhouette cones that approximate concave surfaces when combined with conventional visual hull reconstruction. The approach eliminates protuberant distortions characteristic of conventional IBVH methods while maintaining computational efficiency through image-space operations.  

It should be noted that this summary provides an overview of the invention and is not intended to limit its scope. The detailed description that follows will provide complete disclosure of the inventive concepts, implementations, and advantages.  

## DETAILED DESCRIPTION  

The following detailed description presents the concave object modeling invention with reference to specific embodiments. The invention provides solutions to the fundamental limitations of conventional image-based visual hull (IBVH) techniques in reconstructing concave surfaces and object bottoms through novel image acquisition and processing methods.  

### Concave Object Modeling  

The disclosed concave object modeling approach extends conventional IBVH techniques by incorporating virtual camera perspectives that capture concave surface information. While standard IBVH methods generate visual hulls through intersection of silhouette cones from external viewpoints, the present invention introduces internal virtual viewpoints that enable concave surface approximation.  

### Reconstructing an Image of an Object  

Object reconstruction begins with acquiring multiple reference images from different viewpoints using the specialized image acquisition platform. Each image undergoes segmentation to separate object silhouettes from background. For concave objects, additional specialized images are captured from viewpoints looking into concave regions. These images form the basis for generating virtual inside-out perspectives during reconstruction.  

### Rendering Results of Conventional Processing Methods  

Conventional IBVH processing produces characteristic distortions when applied to concave objects. As shown in exemplary renderings, concave regions appear as convex protrusions due to the visual hull's inherent convexity. Object bottoms similarly exhibit improper conical projections when bottom reference images are unavailable or misaligned. These artifacts significantly degrade reconstruction quality for objects with important concave features or bottom details.  

### Disclosed Approach of Concave Object Modeling  

The disclosed approach overcomes these limitations through virtual camera techniques that mathematically model concave surfaces. By generating virtual images from within concave regions and processing them as negative silhouette cones, the method effectively "carves out" concave features from the visual hull. This approach maintains the computational efficiency of conventional IBVH while dramatically improving reconstruction accuracy.  

### Virtual Camera and Images  

Virtual cameras are mathematically positioned within concave regions to generate synthetic inside-out views. These virtual cameras derive their parameters from corresponding real reference cameras through symmetrical transformations about the mirror plane. The resulting virtual images undergo the same silhouette processing as real images but contribute inverted geometry to the visual hull calculation.  

### Texture Information  

Texture mapping in the disclosed method utilizes weighted blending from multiple reference views. For concave regions, texture sampling prioritizes views that best capture the concave surface details. The method automatically adjusts texture projection to account for corrected surface geometry in concave areas.  

### Block Diagram of Computing Device  

The concave object modeling processing may be implemented on a computing device comprising several key components. A processor executes instructions stored in system memory via a memory bus. Multiple levels of caching (L1, L2, L3) optimize memory access patterns for the computationally intensive reconstruction tasks.  

### Pre-processing Module  

The pre-processing module handles initial image segmentation and silhouette extraction. It identifies concave regions in reference images and prepares them for virtual image synthesis. The module also performs necessary image transformations to align all views within a common reference frame.  

### Virtual-image Synthesis Module  

This module generates virtual camera parameters and corresponding virtual images for concave surface reconstruction. It calculates virtual camera positions through mirror symmetry transformations and synthesizes appropriate inside-out views of concave regions. The module ensures all virtual images maintain proper geometric relationships with real reference images.  

### Point Relocation Module  

During rendering, the point relocation module adjusts three-dimensional coordinates to properly reconstruct concave surfaces. It modifies ray-surface intersection calculations to account for negative silhouette cones from virtual images, effectively "pushing" points into proper concave positions. The module operates entirely in image space to maintain computational efficiency.  

### Processor  

The processor executes the reconstruction algorithms across multiple cores. Each core contains registers for temporary data storage during computation. The processor architecture is optimized for the parallel processing requirements of visual hull reconstruction.  

### System Memory  

System memory stores the operating system, concave object modeling application, and associated data. The memory organization supports rapid access to reference images, silhouette data, and intermediate reconstruction results.  

### Memory Bus  

The memory bus provides high-bandwidth communication between processor, memory, and other system components. Its architecture minimizes latency for memory-intensive reconstruction tasks.  

### Levels of Caching  

Multiple cache levels (L1, L2, L3) optimize memory access patterns. The caching hierarchy is particularly important for silhouette processing and ray intersection calculations that exhibit spatial locality.  

### Processor Core  

Each processor core contains arithmetic logic units optimized for the matrix operations prevalent in visual hull reconstruction. The cores support simultaneous multithreading to maximize utilization during reconstruction.  

### Registers  

Processor registers store intermediate values during geometric calculations. Their organization supports efficient execution of the specialized reconstruction algorithms.  

### Memory Controller  

The memory controller manages data flow between processor and system memory. It prioritizes memory accesses for time-critical reconstruction tasks.  

### Operating System  

The operating system provides necessary services for the concave object modeling application, including memory management, process scheduling, and I/O operations. It includes optimizations for real-time reconstruction performance.  

### Applications  

In addition to the concave object modeling application, the system may include supporting applications for camera calibration, image editing, and visualization. These applications integrate seamlessly with the core reconstruction algorithms.  

### Program Data  

Program data includes reference images, calibration parameters, silhouette masks, and reconstruction results. The data organization facilitates efficient access during all processing stages.  

### Concave Object Modeling Processing Application  

This specialized application implements the complete reconstruction pipeline from image acquisition to final rendering. It coordinates all processing modules and manages data flow through the system.  

### Reference Image Data  

Reference images are stored in optimized formats that balance quality and processing efficiency. The storage scheme supports rapid access during silhouette processing and texture mapping.  

### Bus/Interface Controller  

The bus/interface controller manages communication with peripheral devices including cameras and display outputs. It ensures timely data transfer for real-time reconstruction applications.  

### Data Storage Devices  

The system incorporates various storage devices including solid-state drives for reference image storage and reconstruction results. The storage hierarchy is optimized for the data access patterns of visual hull processing.  

### Removable Storage Devices  

Removable storage devices facilitate transfer of reference images and reconstruction results. Their interfaces support high-speed data transfer for efficient workflow.  

### Non-removable Storage Devices  

Non-removable storage provides persistent storage for the operating system, applications, and frequently used reference data. Its capacity and performance characteristics match reconstruction requirements.  

### Computer Storage Media  

The system utilizes various computer storage media including magnetic and optical formats. Media selection balances capacity, performance, and cost for reconstruction applications.  

### Interface Bus  

The interface bus connects all system components with appropriate bandwidth and latency characteristics. Its architecture supports the parallel processing requirements of visual hull reconstruction.  

### Output Devices  

Output devices include high-resolution displays for visualizing reconstruction results. Their specifications match the quality requirements of professional reconstruction applications.  

### Graphics Processing Unit  

A dedicated GPU accelerates rendering operations including texture mapping and final image synthesis. Its parallel architecture is ideal for visual hull rendering tasks.  

### Audio Processing Unit  

While primarily focused on visual reconstruction, the system includes audio processing capabilities for multimedia applications involving both visual and audio reconstruction.  

### A/V Ports  

Audio/video ports connect the system to capture and display devices. Their specifications support professional-quality input and output for reconstruction applications.  

### Peripheral Interfaces  

Peripheral interfaces connect additional input/output devices as needed for specific reconstruction scenarios. Their flexibility accommodates various camera configurations and display setups.  

### Serial Interface Controller  

The serial interface controller manages communication with devices using serial protocols. It supports various standard and proprietary serial interfaces used in reconstruction equipment.  

### Parallel Interface Controller  

For high-bandwidth device communication, the parallel interface controller manages data transfer to compatible peripherals. Its performance characteristics match reconstruction requirements.  

### I/O Ports  

Input/output ports provide physical connections for reconstruction equipment. Their arrangement facilitates efficient system configuration for different acquisition setups.  

### Communication Device  

The communication device enables network connectivity for distributed reconstruction applications. Its performance supports real-time collaboration in reconstruction projects.  

### Network Controller  

The network controller manages data flow over local and wide area networks. It implements protocols optimized for transferring large reconstruction datasets.  

### Communication Ports  

Communication ports provide physical network connections with appropriate bandwidth for reconstruction applications. Their configuration supports various network topologies.  

### Concave Object Modeling Processing Application  

The concave object modeling processing application implements the complete reconstruction pipeline. It integrates all specialized modules while providing a unified user interface for reconstruction tasks.  

### Silhouette Processing Module  

This module extracts object silhouettes from reference images using advanced segmentation algorithms. It handles challenging cases including transparent objects and complex backgrounds.  

### Virtual-image Synthesis Module  

Building on earlier description, this module additionally optimizes virtual camera placement for maximum concave surface coverage. It automatically determines optimal virtual viewpoints based on object geometry.  

### IBVH Technique  

The core IBVH technique is enhanced to incorporate virtual images in the visual hull calculation. The improved algorithm maintains the efficiency of conventional IBVH while supporting concave reconstruction.  

### Shape-from-Silhouette 3D Reconstruction  

The shape-from-silhouette approach is extended to handle both positive silhouette cones from real images and negative silhouette cones from virtual images. This combination enables accurate concave surface modeling.  

### Silhouette Cone  

The invention introduces the concept of negative silhouette cones generated from virtual inside-out images. These cones subtract volume from the visual hull to create concave features.  

### Image-Based Visual Hull Process  

The standard IBVH process is modified to incorporate virtual image processing. The enhanced process maintains all advantages of the original while overcoming its concave reconstruction limitations.  

### Visual Hull  

The resulting visual hull accurately represents both convex and concave surface features through the combination of real and virtual silhouette cones. This represents a significant improvement over conventional convex visual hulls.  

### Eliminating Protuberant Shaped Distortion  

The method systematically eliminates characteristic protuberant distortions through virtual image processing. Concave regions are reconstructed without artificial convex projections.  

### Determining 3D Coordinates of Concave Region Boundary  

Specialized algorithms determine concave boundary coordinates by combining real and virtual silhouette information. The calculations remain in image space for computational efficiency.  

### Image Segmentation  

Advanced segmentation techniques identify concave regions in reference images. The segmentation results guide virtual camera placement and subsequent processing.  

### Projecting Positions on 2D Reference Picture Back to 3D Space  

The invention improves back-projection accuracy for concave regions by incorporating virtual image constraints. This ensures proper three-dimensional positioning of concave surface points.  

### Choosing Position for Virtual Camera  

Optimal virtual camera positions are determined through analysis of concave region geometry. The positioning algorithm maximizes surface coverage while maintaining computational tractability.  

### Determining Camera Parameters of Virtual Camera  

Virtual camera parameters are derived through symmetrical transformations of real camera parameters. This ensures geometric consistency between real and virtual imaging systems.  

### Generating Virtual Inside-out Image  

The virtual image generation process synthesizes inside-out views of concave regions. These images provide the critical information needed for accurate concave surface reconstruction.