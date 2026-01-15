## FIELD OF THE INVENTION

- relate to multi-view video coding and decoding

The present invention relates to systems and methods for the efficient coding and decoding of multi-view video sequences, particularly in the context of three-dimensional video communication and immersive display technologies. It encompasses the architectural design, signal processing, and bitstream syntax required to enable parallel decoding of multiple spatially distinct video views while preserving high compression efficiency and maintaining compatibility with existing video coding standards. The invention is particularly directed to the implementation of parallel decoding pipelines in multi-view video decoders, where each view is processed independently yet coherently with respect to inter-view dependencies, enabling real-time rendering of stereoscopic or autostereoscopic content on consumer-grade hardware. The invention further pertains to the signaling mechanisms that constrain and describe the inter-view prediction relationships in a manner that permits synchronized, non-sequential decoding of multiple views without compromising the integrity of motion compensation or reconstruction. This technology finds application in 3D television broadcasting, immersive teleconferencing, free-viewpoint video streaming, and head-motion parallax-enabled displays, where simultaneous decoding of multiple camera perspectives is essential for delivering a natural three-dimensional visual experience.

## BACKGROUND OF THE INVENTION

- introduce 3D video communication and entertainment services

Three-dimensional video communication and entertainment services have emerged as a transformative frontier in multimedia technology, enabling users to perceive depth and spatial relationships in a manner that closely mimics natural human vision. These services encompass a broad spectrum of applications, including stereoscopic 3D television, immersive telepresence systems, and interactive free-viewpoint video platforms, all of which rely on the capture, transmission, and rendering of multiple camera viewpoints to reconstruct a three-dimensional scene. Unlike traditional two-dimensional video, these systems require the simultaneous or rapidly alternating display of multiple perspectives, each corresponding to a slightly different spatial position relative to the scene. The fidelity of the three-dimensional experience is directly proportional to the number of views rendered and the precision with which they are synchronized, particularly when head-motion parallax is supported, allowing viewers to perceive occluded objects by shifting their gaze. The successful deployment of such systems demands not only advanced capture and display hardware but also highly efficient coding and decoding architectures capable of handling the substantial data burden imposed by multiple video streams.

- describe 3D display characteristics

Modern three-dimensional displays are designed to present distinct image streams to each eye or to multiple viewing zones, enabling depth perception without the need for auxiliary viewing devices such as shutter glasses. Autostereoscopic displays, in particular, achieve this by employing optical elements such as lenticular lenses or parallax barriers that direct different portions of the image to different viewing angles. These displays are capable of supporting multiple simultaneous viewers, each experiencing a unique perspective based on their physical location relative to the screen. The quality of the three-dimensional illusion is enhanced when the number of displayed views is increased, as this allows for smoother transitions during head movement and reduces the perception of image discontinuities. However, increasing the number of views also exponentially increases the computational and bandwidth requirements for decoding and rendering, necessitating novel approaches to video processing that balance visual fidelity with system scalability.

- explain head motion-parallax feature

The head motion-parallax feature is a critical perceptual cue in three-dimensional vision, wherein the apparent position of objects within a scene shifts relative to one another as the viewer changes their viewing angle. This phenomenon, fundamental to human depth perception, allows viewers to infer the spatial layout of a scene by observing how objects occlude or reveal one another during lateral or vertical head movements. In three-dimensional video systems, the faithful reproduction of head motion-parallax requires the display of multiple, spatially offset views of the same scene, each corresponding to a different observer position. When the viewer moves, the system must seamlessly switch between or interpolate among these views to maintain the illusion of depth. Without sufficient view density, the parallax effect becomes discontinuous or delayed, resulting in visual discomfort or a loss of immersion. Consequently, the ability to decode and render multiple views in real time, with minimal latency and precise synchronization, is paramount to delivering a compelling three-dimensional experience.

- introduce Multi-View Video Coding (MVC) standard

The Multi-View Video Coding (MVC) standard, an extension of the H.264/AVC video compression framework, was developed to address the unique challenges of encoding multiple correlated video streams captured from different spatial viewpoints. By exploiting the strong pixel-level redundancies between adjacent views—arising from their shared scene content and temporal coherence—MVC enables significant compression gains over independent coding of each view. The standard introduces novel syntax elements and prediction structures that allow pictures from one view to be used as reference frames during the encoding of another, provided they correspond to the same temporal instant. This inter-view prediction mechanism substantially reduces bitrates while preserving visual quality, making the transmission of multi-view content feasible over bandwidth-constrained networks. The MVC standard also retains backward compatibility with conventional H.264/AVC decoders by designating one view as a base layer, which can be decoded independently using legacy equipment.

- describe inter-view dependencies in MVC

In MVC, inter-view dependencies are established through the use of reference picture lists that include pictures from other views, enabling motion compensation and residual prediction across spatially distinct camera perspectives. These dependencies are formally defined in the sequence parameter set and may vary dynamically across temporal instances, allowing for flexible prediction structures that adapt to scene content and viewing geometry. However, the presence of such dependencies introduces a fundamental constraint on parallel processing: a picture in one view may require the complete reconstruction of one or more pictures in another view before its own decoding can proceed. This sequential dependency chain can severely limit the potential for parallelization, particularly in systems requiring the simultaneous rendering of multiple views, as the decoder must wait for upstream views to complete before initiating downstream processing. As a result, even with multiple processing cores, the effective throughput of the decoder is often bottlenecked by the longest dependency path, leading to increased latency and reduced real-time performance.

- explain complexity problems and parallelism issues

The complexity of decoding multi-view video sequences is compounded by the need to manage large decoded picture buffers, synchronize output across multiple views, and resolve inter-view motion vectors with high precision. Traditional MVC decoders, operating in a sequential manner, are unable to exploit the inherent parallelism offered by modern multi-core architectures, resulting in suboptimal utilization of available computational resources. The sequential nature of inter-view prediction forces the decoder to process views in a strict temporal and spatial order, preventing independent threads from operating concurrently on different views. This limitation becomes particularly acute in applications such as 3D television broadcasting, where the simultaneous display of ten or more views is required to support wide viewing angles and smooth head-motion parallax. Under such conditions, the computational load may exceed the capacity of consumer-grade hardware, rendering real-time decoding impractical without architectural innovation.

- illustrate decoding issues in 3D-TV systems

In 3D-TV systems, the requirement to decode and render multiple views simultaneously imposes stringent demands on both memory bandwidth and processing throughput. The decoded picture buffer must retain not only the reference pictures for temporal prediction within each view but also the inter-view reference pictures necessary for motion compensation across views. This results in a buffer size that scales multiplicatively with the number of views and the depth of the hierarchical prediction structure. Furthermore, the synchronization of output timing across views must be maintained with sub-frame precision to prevent visual artifacts such as flicker, ghosting, or temporal misalignment. Without careful coordination, the asynchronous completion of decoding tasks across views can lead to frame drops, stuttering, or delayed rendering, all of which degrade the viewer experience. These issues are exacerbated in mobile and embedded environments where power and thermal constraints limit the scalability of decoding hardware.

- motivate need for parallel decoding of separate views

The need for parallel decoding of separate views arises from the fundamental mismatch between the computational demands of multi-view video and the architectural capabilities of modern decoding platforms. While multi-core processors and dedicated video accelerators are now commonplace, their potential remains largely untapped in conventional MVC decoders due to the sequential nature of inter-view dependencies. By introducing systematic constraints on the reference picture availability and explicitly signaling the minimum decoding delay between views, it becomes possible to partition the decoding workload across independent processing units, each responsible for a distinct view. This approach enables true parallelism, reducing end-to-end latency and improving throughput without sacrificing coding efficiency. Such a paradigm shift is essential for the widespread adoption of immersive 3D services in consumer electronics, broadcast infrastructure, and mobile streaming platforms, where real-time performance and energy efficiency are paramount.

## SUMMARY OF THE INVENTION

- introduce parallel decoder implementation for different views

The present invention introduces a novel parallel decoder implementation for multi-view video sequences that enables the simultaneous and independent decoding of multiple views while preserving the compression efficiency of inter-view prediction. This is achieved through a system architecture in which each view is assigned to a dedicated decoding pipeline, with explicit signaling mechanisms that define the minimum temporal delay required between the completion of reference picture reconstruction in one view and the initiation of decoding in another. By constraining the available reference area for inter-view motion compensation to a predetermined subset of macroblocks, the invention eliminates data dependencies that would otherwise prevent parallel execution. This allows multiple decoding threads to operate concurrently, each progressing through its assigned view without waiting for the full reconstruction of other views, thereby significantly reducing decoding latency and improving overall system throughput.

- describe encoding constraints for parallel decoding

The invention further defines a set of encoding constraints that ensure the bitstream is structured to support parallel decoding without compromising visual fidelity. These constraints include the limitation of inter-view motion vectors to reference only those macroblocks that have been fully reconstructed and are available within a specified delay window. The encoder is required to analyze the spatial and temporal relationships between views and to encode each picture such that the motion compensation process for any macroblock in a dependent view relies exclusively on reconstruction data from a subset of macroblocks in the reference view that are guaranteed to be available prior to a defined decoding offset. This systematic restriction is communicated to the decoder via supplemental enhancement information (SEI) messages, which specify the allowable reference range and the initial delay between views, ensuring that the decoder can safely initiate parallel processing without risking invalid predictions or reconstruction errors.

- explain macroblock delay signaling

A key innovation of the invention lies in the signaling of macroblock-level delay parameters that govern the timing of parallel decoding operations. These parameters, encoded within the bitstream as syntax elements such as pds_block_size and pds_initial_delay, indicate the number of macroblock rows that must be decoded in a reference view before decoding can commence in a dependent view. The pds_block_size defines the granularity of the delay constraint, while pds_initial_delay specifies the number of such blocks that must be completed prior to initiating decoding in the dependent view. These values are transmitted in a parallel decoding information SEI message, which is associated with the relevant video sequence and may be overridden on a per-slice basis using the parallelly_decodable_slice_flag. This fine-grained signaling enables the encoder to adapt the delay constraints dynamically based on scene content, motion complexity, and available bandwidth, ensuring optimal trade-offs between parallelism and coding efficiency.

- summarize advantages of the invention

The invention offers a multitude of advantages over conventional multi-view decoding architectures. First, it enables true parallel decoding of multiple views, dramatically reducing end-to-end latency and enabling real-time rendering on resource-constrained devices. Second, it preserves the full compression efficiency of inter-view prediction, avoiding the bitrate penalties associated with simulcast approaches. Third, it maintains backward compatibility with existing H.264/AVC decoders by ensuring that the base view remains fully decodable without reference to the SEI signaling. Fourth, it supports scalable deployment across varying hardware configurations, from single-core embedded systems to multi-threaded high-end rendering platforms. Finally, the invention facilitates seamless integration with existing media frameworks, including adaptive streaming protocols and media gateways, by embedding all necessary signaling within standardized NAL unit structures and SEI message formats, thereby enabling widespread adoption without requiring modifications to network infrastructure or client software.

## BRIEF DESCRIPTION OF THE DRAWINGS

- describe FIG. 1 conventional sample prediction chain

Figure 1 illustrates a conventional prediction chain in a multi-view video coding system, wherein the decoding of a macroblock in a dependent view is contingent upon the complete reconstruction of all preceding macroblocks in the reference view. The diagram depicts a temporal sequence of frames across two views, with arrows indicating the direction of inter-view motion compensation. Each macroblock in the dependent view references a corresponding region in the reference view, but due to the sequential nature of the decoding process, the entire reference frame must be processed before any macroblock in the dependent view can begin. This results in a linear dependency chain that prevents parallel execution and imposes significant latency on the decoding pipeline.

- describe FIG. 2 system overview diagram

Figure 2 presents a system overview diagram of the parallel multi-view decoding architecture according to the invention. The diagram shows multiple independent decoding pipelines, each assigned to a distinct video view, with a central signaling module that transmits parallel decoding parameters via SEI messages. The encoder generates a bitstream that includes both VCL NAL units and non-VCL SEI NAL units containing the pds_block_size, pds_initial_delay, and parallelly_decodable_slice_flag syntax elements. The receiver parses these parameters and configures each decoding thread to initiate processing only after the required number of macroblock rows have been reconstructed in the reference view. The output of each pipeline is synchronized and passed to a renderer for simultaneous display, enabling real-time, low-latency rendering of multiple views.

- describe remaining figures

The remaining figures illustrate the detailed operation of the parallel decoding mechanism, including the representation of macroblock dependency regions, the signaling of available reference areas, the modified deblocking and sub-pixel interpolation processes, and the adaptation of entropy coding for parallel contexts. Additional figures depict the implementation of the invention on electronic devices, the organization of NAL units and SEI messages within the bitstream, and the software architecture of the decoder implementation. Each figure provides a visual representation of the technical components and data flow that enable the invention’s core functionality, reinforcing the structural and operational novelty of the disclosed system.

## DETAILED DESCRIPTION OF VARIOUS EMBODIMENTS

- introduce parallel decoder implementation

The parallel decoder implementation of the invention comprises multiple independent decoding pipelines, each dedicated to the processing of a single video view within a multi-view sequence. Each pipeline is equipped with its own motion compensation unit, inverse transform module, and deblocking filter, allowing for concurrent execution without interference. The pipelines are coordinated through a central control unit that receives signaling parameters from the bitstream, specifically the parallel decoding information SEI message, which defines the minimum delay between the completion of reference picture reconstruction and the initiation of decoding in dependent views. This architecture eliminates the sequential bottleneck inherent in conventional MVC decoders, enabling true parallelism and significantly improving real-time performance.

- describe multimedia communications system

The multimedia communications system in which the invention operates includes a multi-view video source, an encoder that applies the parallel decoding constraints, a transmission network, a media gateway capable of bitstream adaptation, and a receiving device equipped with the parallel decoder. The encoder generates a bitstream compliant with the H.264/AVC standard, augmented with SEI messages that signal the parallel decoding parameters. The media gateway may selectively filter or transcode the bitstream based on client capabilities, preserving the parallel decoding constraints in the adapted stream. The receiving device, whether a set-top box, mobile terminal, or immersive display system, parses the SEI messages and configures its decoding pipelines accordingly, ensuring that the parallel decoding mechanism functions correctly regardless of the transmission path.

- illustrate data source and encoder

The data source comprises an array of synchronized cameras capturing the same scene from multiple spatial positions, each producing a temporal sequence of video frames. The encoder processes these frames using a modified H.264/AVC codec that incorporates inter-view prediction with constrained reference areas. The encoder analyzes the motion vectors and spatial relationships between views to determine the minimum number of macroblock rows that must be decoded in a reference view before a dependent view can begin processing. It then encodes each picture with motion vectors restricted to the allowable reference region and embeds the corresponding delay parameters into the bitstream via SEI messages, ensuring that the decoder can reconstruct the scene without violating the parallelism constraints.

- explain encoding process

The encoding process begins with the temporal and inter-view prediction of each picture, followed by the calculation of motion vectors and residual data. For each macroblock in a dependent view, the encoder verifies that its motion vector points to a region in the reference view that is guaranteed to be fully reconstructed within the specified delay window. If a motion vector would reference an unavailable region, the encoder modifies the prediction structure by selecting an alternative reference or adjusting the motion vector to remain within the permitted area. The encoder then generates a parallel decoding information SEI message that includes the pds_block_size, pds_initial_delay, and pds_parameters_present_flag, which collectively define the constraints under which parallel decoding is permissible.

- describe storage and sender

The encoded bitstream is stored in a media container or transmitted over a network using a protocol stack that supports NAL unit segmentation and SEI message embedding. The sender, whether a server, media gateway, or streaming device, ensures that the SEI messages are preserved during any adaptation or transcoding process. The bitstream may be segmented into operation points based on view and temporal scalability, with each operation point containing the necessary signaling to support parallel decoding. The sender may also apply priority-based filtering to reduce bandwidth, but only if the resulting bitstream continues to satisfy the parallel decoding constraints.

- illustrate communication protocol stack

The communication protocol stack includes layers for transport, network, and application, with the video bitstream encapsulated in NAL units that are transmitted over UDP, RTP, or HTTP-based streaming protocols. The SEI messages are carried within non-VCL NAL units and are processed by the decoder prior to initiating any decoding pipeline. The protocol stack ensures that the SEI messages arrive before the corresponding VCL NAL units, allowing the decoder to configure its parallel pipelines in advance. The stack is designed to be compatible with existing media gateways and does not require modifications to core network infrastructure.

- describe sender and gateway

The sender is responsible for generating the bitstream with the appropriate parallel decoding signaling, while the gateway may adapt the bitstream for network conditions or client capabilities. The gateway is configured to preserve the parallel decoding information SEI messages during any form of transcoding, filtering, or rate adaptation. If a view is removed from the bitstream, the gateway recalculates the remaining dependencies and updates the SEI messages accordingly to maintain the integrity of the parallel decoding constraints.

- explain receiver and decoder

The receiver captures the incoming bitstream and parses the NAL units to extract the SEI messages containing the parallel decoding parameters. Upon detecting the pds_parameters_present_flag, the decoder initializes multiple decoding pipelines, each assigned to a distinct view. The decoder uses the pds_initial_delay and pds_block_size values to determine when each pipeline may begin processing, ensuring that the required reference macroblocks have been fully reconstructed. The decoder synchronizes the output of all pipelines to ensure temporal alignment and presents the reconstructed frames to the renderer.

- describe renderer

The renderer receives the decoded frames from each pipeline and combines them into a unified three-dimensional display output. It applies spatial interpolation, depth mapping, and optical correction to generate the final image presented to the viewer. The renderer is synchronized with the decoder to ensure that frames from all views are displayed simultaneously, preserving the integrity of the head-motion parallax effect. The renderer may operate on a display with autostereoscopic capabilities, supporting multiple viewing zones without the need for auxiliary viewing devices.

- motivate scalability

The invention is inherently scalable, as the number of parallel decoding pipelines can be increased or decreased based on the number of views in the bitstream and the computational resources available on the receiving device. The signaling mechanism allows the decoder to adapt its architecture dynamically, enabling deployment on devices ranging from low-power mobile phones to high-end immersive displays. The scalability is further enhanced by the ability to selectively decode subsets of views, as defined by operation points signaled in the VSSEI message, without compromising the parallelism of the remaining views.

- list transmission technologies

Transmission technologies supported by the invention include wired broadband networks, cellular networks, satellite links, and local area wireless systems such as Wi-Fi and 5G. The bitstream may be delivered via unicast, multicast, or broadcast protocols, with the parallel decoding constraints preserved in all cases. The invention is compatible with adaptive bitrate streaming standards such as DASH and HLS, allowing seamless adaptation to varying network conditions.

- list communication devices

Communication devices capable of implementing the invention include set-top boxes, smart televisions, mobile smartphones, tablets, virtual reality headsets, automotive infotainment systems, and immersive teleconferencing terminals. All such devices may be equipped with multi-core processors and dedicated video decoding hardware that can execute the parallel decoding pipelines in parallel, enabling real-time rendering of multi-view content.

- illustrate representation of frames

The frames in the bitstream are organized in time-first coding order, with all views corresponding to a single temporal instant grouped into an access unit. Each access unit contains NAL units for all views, with the base view appearing first in decoding order. The parallel decoding constraints are applied on a per-access-unit basis, ensuring that the delay parameters remain consistent across the entire temporal sequence.

- describe parallel decoding of views

Parallel decoding of views is enabled by restricting the inter-view motion compensation to a limited reference area, as defined by the pds_block_size and pds_initial_delay parameters. Each decoding pipeline begins processing its assigned view only after the required number of macroblock rows have been reconstructed in the reference view. This ensures that all necessary reference data are available without requiring the full reconstruction of the reference view, thereby permitting concurrent execution of multiple pipelines.

- explain decoding process for two views

In the case of two views, the decoder initializes two pipelines, one for each view. The first pipeline decodes the base view without constraints. The second pipeline waits for the pds_initial_delay value to elapse, during which the first pipeline decodes the required number of macroblock rows. Once the delay condition is satisfied, the second pipeline initiates decoding, using only the macroblocks that have been reconstructed within the allowable reference area. The two pipelines proceed independently thereafter, with no further synchronization required.

- illustrate decoding process

The decoding process is illustrated through a sequence of diagrams showing the progression of macroblock reconstruction across time and view. The diagrams depict the available reference area expanding incrementally as each macroblock row is completed, and the dependent view initiating decoding only after the threshold is reached. The diagrams also show the motion vectors constrained within the permitted region, ensuring that no invalid references are made.

- describe WAIT state

The WAIT state is a temporary condition in which a decoding pipeline is suspended until the required number of macroblock rows have been reconstructed in the reference view. During this state, the pipeline remains idle but retains its context, including the decoded picture buffer and motion vector history. Once the delay condition is met, the pipeline transitions to the DECODE state and resumes processing without interruption.

- explain decoding operation

The decoding operation follows the standard H.264/AVC procedures for motion compensation, inverse transform, and intra prediction, with the additional constraint that inter-view motion vectors are validated against the available reference area. If a motion vector points outside the permitted region, the decoder substitutes an alternative reference or applies a fallback prediction mode. The operation is performed independently for each view, with no cross-pipeline dependencies beyond the signaling of the delay parameters.

- describe notification process

The notification process is triggered when the required number of macroblock rows have been reconstructed in the reference view. A signal is sent from the reference pipeline to the dependent pipeline, indicating that the necessary data are available. This notification is implemented through a shared memory barrier or a thread synchronization primitive, ensuring that the dependent pipeline does not proceed until the reference data are fully written and accessible.

- explain parallel implementation

The parallel implementation leverages multi-threading or multi-core processing to execute each view’s decoding pipeline independently. The operating system or hardware scheduler assigns each pipeline to a separate processing unit, allowing for true concurrent execution. The signaling mechanism ensures that no data race conditions occur, as the dependency constraints are enforced at the bitstream level rather than through runtime synchronization.

- describe delay and synchronization overhead

The delay and synchronization overhead introduced by the invention is minimal, as the pds_initial_delay parameter is optimized to balance parallelism with coding efficiency. The overhead is typically less than one macroblock row’s worth of decoding time, which corresponds to a fraction of a millisecond in standard video frame rates. This small delay is more than offset by the elimination of sequential bottlenecks, resulting in a net reduction in end-to-end latency.

- motivate signaling macroblock delay

Signaling the macroblock delay is motivated by the need to enable parallel decoding without requiring the decoder to perform complex dependency analysis at runtime. By encoding the constraints directly into the bitstream, the invention ensures that the decoder can operate with deterministic behavior, regardless of the complexity of the inter-view prediction structure. This eliminates the need for heuristic algorithms or adaptive scheduling, simplifying implementation and improving reliability.

- define syntax elements

The syntax elements defined by the invention include pds_parameters_present_flag, which indicates the presence of parallel decoding parameters; pds_block_size, which defines the size of the macroblock delay unit; pds_initial_delay, which specifies the number of delay units required before decoding may proceed; and parallelly_decodable_slice_flag, which enables per-slice override of the global delay parameters. These elements are encoded within the parallel decoding information SEI message and are parsed by the decoder prior to initiating any decoding pipeline.

- explain available reference area

The available reference area is the region in a reference view from which motion vectors in a dependent view are permitted to derive their prediction data. This area is defined by the pds_block_size and pds_initial_delay parameters and is calculated as the number of macroblock rows that have been fully reconstructed prior to the initiation of decoding in the dependent view. The available reference area equation ensures that the motion compensation process remains valid and that no reconstruction errors occur due to incomplete reference data.

- describe pds_block_size and pds_initial_delay

The pds_block_size parameter defines the granularity of the delay constraint, specifying the number of macroblocks per row that constitute a single delay unit. The pds_initial_delay parameter specifies the number of such units that must be completed in the reference view before decoding may begin in the dependent view. Together, these parameters allow the encoder to fine-tune the trade-off between parallelism and coding efficiency, enabling optimal performance across diverse content types and hardware platforms.

- illustrate sample decoding process

A sample decoding process is illustrated through a step-by-step sequence showing the reconstruction of macroblocks in a reference view and the subsequent initiation of decoding in a dependent view. The sequence demonstrates how the available reference area expands incrementally and how the dependent view begins decoding only after the pds_initial_delay threshold is reached, with motion vectors constrained to the permitted region.

- describe pds_parameters_present_flag

The pds_parameters_present_flag is a single-bit indicator that signals whether the parallel decoding parameters are present in the bitstream. If the flag is set to one, the decoder expects the pds_block_size, pds_initial_delay, and other related syntax elements to follow. If the flag is zero, the decoder reverts to conventional sequential decoding behavior, ensuring backward compatibility with legacy systems.

- explain fixed_pds_for_all_sequence_flag

The fixed_pds_for_all_sequence_flag is a binary indicator that determines whether the parallel decoding parameters apply uniformly across the entire video sequence or may vary on a per-slice basis. If the flag is set to one, the parameters are fixed for the entire sequence, simplifying decoder implementation. If the flag is zero, the parameters may be overridden by the parallelly_decodable_slice_flag in individual slices, allowing for dynamic adaptation to changing scene content.

- describe parallelly_decodable_slice_flag

The parallelly_decodable_slice_flag is a per-slice indicator that enables or disables the parallel decoding constraints for individual slices within a picture. When set to one, the slice is subject to the global delay parameters. When set to zero, the slice may be decoded independently, even if the reference area is not fully available, allowing for localized flexibility in regions with minimal inter-view dependency.

- explain available_reference_area equation

The available_reference_area equation is defined as the product of pds_block_size and pds_initial_delay, yielding the total number of macroblock rows that must be reconstructed in the reference view before decoding may proceed in the dependent view. This equation ensures that the decoder can compute the required delay without additional parsing or runtime analysis, enabling deterministic and efficient parallel execution.

- discuss adaptive deblocking and sub-pixel interpolation

Adaptive deblocking and sub-pixel interpolation are modified to account for the constrained reference area. The deblocking filter is applied only to macroblock boundaries that lie within the available reference area, preventing artifacts from being introduced by incomplete reconstruction. Sub-pixel interpolation is restricted to the same region, ensuring that interpolated pixels are derived from valid reference data. These modifications preserve visual quality while maintaining the integrity of the parallel decoding mechanism.

- describe sliding deblocking approach

The sliding deblocking approach applies the deblocking filter incrementally as each macroblock row becomes available, rather than waiting for the entire picture to be reconstructed. This approach reduces memory latency and allows the filter to operate in parallel with the decoding pipeline, further improving system throughput.

- illustrate decoding process for first and second views

The decoding process for the first and second views is illustrated through a timeline diagram showing the progression of macroblock reconstruction and the initiation of decoding in the second view after the pds_initial_delay threshold is reached. The diagram demonstrates the absence of data dependencies between the pipelines after the initial delay, confirming the validity of the parallel implementation.

- detail filtering of macroblock boundaries

The filtering of macroblock boundaries is performed using a modified deblocking algorithm that considers only the available reference area when determining edge strength and filter strength. This prevents the introduction of artifacts at boundaries that extend beyond the permitted reference region, ensuring that the reconstructed image remains visually consistent.

- describe modified deblocking operation

The modified deblocking operation applies the standard H.264/AVC deblocking filter but restricts its application to macroblock edges that lie entirely within the available reference area. Edges that cross into unverified regions are left unfiltered, and the decoder may apply a fallback smoothing operation to minimize visual discontinuities.

- discuss sub-pixel interpolation

Sub-pixel interpolation is performed using a restricted set of reference pixels that lie within the available reference area. The interpolation kernel is applied only to pixels that are fully supported by reconstructed data, preventing the introduction of artifacts from incomplete or invalid reference values.

- illustrate effect of sub-pixel interpolation

The effect of sub-pixel interpolation is illustrated through a comparison of images decoded with and without the constrained reference area. The images show that the modified interpolation preserves motion accuracy and edge clarity, with no perceptible degradation in visual quality compared to conventional decoding.

- describe padding approach for addressing unavailable pixels

When motion vectors point to regions outside the available reference area, the decoder applies a padding approach, substituting the nearest available pixel values to fill the missing data. This approach ensures that motion compensation can proceed without error, albeit with a slight reduction in prediction accuracy.

- describe limiting reference area approach

The limiting reference area approach enforces a hard boundary on the region from which motion vectors may derive their data. This boundary is defined by the pds_block_size and pds_initial_delay parameters and is validated during the encoding process to ensure that no motion vector exceeds the permitted range.

- discuss degradation of coding efficiency

The degradation of coding efficiency due to the constrained reference area is minimal, typically less than 0.08 dB in objective metrics and imperceptible in subjective evaluations. The trade-off between parallelism and compression efficiency is carefully balanced by the encoder, which selects the optimal delay parameters based on scene content and motion complexity.

- describe arranging view dependencies

View dependencies are arranged such that the base view is decoded first, and dependent views are scheduled to begin decoding only after the required number of macroblock rows have been reconstructed. The dependency graph is encoded into the sequence parameter set and reinforced by the SEI signaling, ensuring that the decoder can construct a valid decoding schedule.

- describe modifying original picture

The original picture is modified during encoding to ensure that motion vectors remain within the available reference area. This may involve re-encoding certain macroblocks with alternative prediction modes or adjusting the motion vector values to conform to the constraints.

- describe utilizing slice groups

Slice groups are utilized to partition pictures into regions with similar dependency characteristics. Slices within a group may share the same parallel decoding parameters, reducing the overhead of signaling and simplifying decoder configuration.

- describe modified raster scan

The modified raster scan orders the decoding of macroblocks such that those with the highest dependency on other views are processed last, ensuring that the available reference area is maximized before critical regions are decoded.

- describe signaling through SEI message syntax

Signaling is achieved through a standardized SEI message syntax that includes the pds_parameters_present_flag, pds_block_size, pds_initial_delay, fixed_pds_for_all_sequence_flag, and parallelly_decodable_slice_flag. These elements are encoded in a single SEI NAL unit that precedes the corresponding VCL NAL units, ensuring that the decoder can configure its pipelines before decoding begins.

- detail NAL unit and bytestream format

The NAL unit format is extended to include the SEI NAL unit type for parallel decoding information, which is carried in the non-VCL portion of the bytestream. The bytestream format retains full compatibility with H.264/AVC, with the SEI messages inserted as optional enhancements that do not affect decoding in legacy systems.

- describe SEI NAL unit and SEI messages

The SEI NAL unit contains one or more SEI messages, each identified by a payload type. The parallel decoding information SEI message is assigned a unique payload type and includes the syntax elements necessary to define the parallel decoding constraints. The SEI messages are transmitted out-of-band and are processed by the decoder prior to any VCL NAL unit decoding.

- discuss user data SEI messages

User data SEI messages are used to carry proprietary or application-specific parameters that may augment the parallel decoding mechanism. These messages are ignored by standard decoders but may be interpreted by advanced implementations to enable additional optimizations.

- describe signaling of parallelly decodable slice parameters

The signaling of parallelly decodable slice parameters is performed on a per-slice basis using the parallelly_decodable_slice_flag. This flag allows the encoder to selectively disable the delay constraints in regions where inter-view dependency is minimal, enabling greater flexibility in complex scenes.

- discuss taking advantage of PDS arrangement

The PDS arrangement allows the decoder to exploit the structured nature of the delay constraints to optimize memory access patterns, prefetch reference data, and schedule decoding tasks efficiently. This results in improved cache utilization and reduced memory bandwidth requirements.

- describe entropy coding arrangements

Entropy coding arrangements are adapted to handle the constrained motion vector ranges by using shorter codewords for values that fall within the available reference area. This reduces the bitrate overhead associated with signaling motion vectors while maintaining compatibility with standard CAVLC and CABAC implementations.

- detail CAVLC and CABAC implementations

The CAVLC and CABAC implementations are modified to use context models that are conditioned on the available reference area. Motion vectors that are constrained to the permitted region are encoded using lower entropy contexts, resulting in more efficient bit allocation and reduced bitrate.

- describe motion vector coding

Motion vector coding is constrained to the available reference area, ensuring that the predicted values fall within a limited range. This allows the use of shorter variable-length codes and reduces the number of bits required to represent each motion vector.

- illustrate ranges for horizontal and vertical components of motion vectors

The ranges for the horizontal and vertical components of motion vectors are illustrated through a diagram showing the permissible region relative to the available reference area. The diagram demonstrates that the motion vector values are bounded by the pds_block_size and pds_initial_delay parameters, ensuring that no vector extends beyond the reconstructed region.

- describe single codeword arrangement

The single codeword arrangement encodes the horizontal and vertical components of the motion vector as a single unit, using a context-adaptive model that is conditioned on the available reference area. This reduces the number of syntax elements and improves coding efficiency.

- describe separate coding of horizontal and vertical components

The separate coding of horizontal and vertical components is employed when the motion vector ranges differ significantly between dimensions. Each component is encoded using its own context model, allowing for more precise bit allocation and improved compression.

- illustrate adapting variable length codes

The adaptation of variable length codes is illustrated through a comparison of code tables used with and without the parallel decoding constraints. The constrained code tables use shorter codes for frequently occurring motion vector values, resulting in a measurable reduction in bitrate.

- describe electronic device implementation

The electronic device implementation includes a multi-core processor, dedicated video decoding hardware, and memory subsystems configured to support parallel decoding pipelines. The device runs a firmware or software decoder that parses the SEI messages and configures the pipelines accordingly.

- detail components of electronic device

The components of the electronic device include a central processing unit, a graphics processing unit, a video decoding accelerator, a memory controller, and an output interface. The video decoding accelerator is optimized to execute the parallel decoding pipelines in parallel, with each pipeline assigned to a separate processing core.

- describe program product implementation

The program product implementation consists of a computer program stored on a non-transitory medium, which, when executed, performs the steps of parsing the SEI messages, configuring the decoding pipelines, and executing the parallel decoding process. The program is compatible with multiple operating systems and hardware architectures.

- discuss computer-readable medium

The computer-readable medium may be a solid-state drive, optical disc, or flash memory, containing the program instructions and data structures necessary to implement the invention. The medium is readable by standard computing devices and may be distributed via digital download or physical media.

- describe software and web implementations

Software implementations are provided as libraries or plugins that integrate with existing media players and streaming clients. Web implementations are delivered as JavaScript modules that execute in a browser environment, enabling parallel multi-view decoding on web-based platforms without requiring native applications.

- discuss rule-based logic and other logic

Rule-based logic is employed to determine the optimal delay parameters based on scene content, motion complexity, and hardware capabilities. Other logic, including machine learning models, may be used to predict the best configuration for a given bitstream, further enhancing the efficiency of the parallel decoding mechanism.

- describe database searching steps

Database searching steps are used to retrieve precomputed delay parameters for common video sequences, reducing the computational overhead of encoding. The database is indexed by scene type, view count, and resolution, allowing for rapid lookup and application of optimal parameters.

- describe correlation steps

Correlation steps are performed during encoding to identify regions of high inter-view similarity, allowing the encoder to prioritize the allocation of delay parameters to those regions where parallelism yields the greatest benefit.

- describe comparison steps

Comparison steps are used to evaluate the trade-off between parallelism and coding efficiency, selecting the delay parameters that minimize the overall bitrate while maximizing decoding throughput.

- describe decision steps

Decision steps are implemented in the encoder to determine whether to apply the parallel decoding constraints on a per-picture or per-slice basis, based on the complexity of the inter-view dependencies and the available bandwidth.

- discuss modifications and variations

Modifications and variations of the invention include extending the signaling mechanism to support three or more views simultaneously, incorporating temporal scalability into the delay constraints, and integrating the mechanism with scalable video coding standards such as SVC.

- describe combining features of embodiments

The features of the various embodiments may be combined in any configuration to suit specific application requirements. For example, the parallel decoding mechanism may be combined with view scalability, temporal scalability, and error resilience features to create a comprehensive multi-view video system that is both efficient and robust.