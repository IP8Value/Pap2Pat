# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to the field of multiview video coding (MVC) and, more specifically, to methods and systems for efficiently compressing and transmitting multiview video content. The invention addresses the challenges associated with the large volume of data in multiview video applications and provides mechanisms to enhance compression efficiency, support view scalability, enable parallel processing, and ensure robustness in transmission.

## BACKGROUND OF THE INVENTION

Three-dimensional (3D) video has gained significant interest due to advancements in acquisition and display technologies. Various applications of 3D video, such as free-viewpoint video, 3D television (3D TV), and immersive teleconferencing, are becoming feasible. Each of these applications has unique requirements and challenges, such as efficient compression, view scalability, and real-time decoding.

### Free-Viewpoint Video

In free-viewpoint video, the viewer can interactively choose their viewpoint in 3D space to observe a real-world scene from preferred perspectives. This requires the system to support efficient switching between different views and the ability to interpolate virtual views when the desired viewpoint is not available.

### 3D Television

3D TV extends traditional 2D TV displays to support 3D rendering. This can be achieved through stereoscopic video or autostereoscopic displays, which allow viewers to experience 3D without special glasses. Advanced autostereoscopic displays support head-motion parallax by decoding and displaying multiple views simultaneously. Efficient parallel processing and minimal memory consumption are crucial for real-time decoding in 3D TV applications.

### Immersive Teleconferencing

Immersive teleconferencing combines interactivity and virtual reality, supporting both free-viewpoint video and 3D TV styles. This application requires robust error resilience and efficient view scalability to accommodate varying network conditions and decoder capabilities.

### Challenges in Multiview Video Coding

The primary challenge in multiview video coding is the efficient compression of the large volume of data. Exploiting the correlation between views, in addition to inter-prediction in monoview coding, is essential for improving compression efficiency. However, this requires efficient memory management and parallel processing capabilities to handle the increased computational load.

### Existing Solutions

Existing coding standards, such as H.264/AVC, provide a foundation for multiview video coding. However, they lack the specific features required to address the unique challenges of multiview video applications. The emerging MVC standard, developed by the Joint Video Team (JVT), aims to fill this gap by introducing mechanisms for view scalability, temporal scalability, and parallel processing.

## SUMMARY OF THE INVENTION

The present invention provides a method and system for multiview video coding (MVC) that addresses the challenges of efficient compression, view scalability, parallel processing, and robustness in transmission. The invention includes the following key features:

1. **Efficient Compression**: The invention utilizes inter-view prediction to exploit the correlation between views, thereby improving compression efficiency. This is achieved through a flexible prediction structure that allows decoded pictures of other views to be used as reference pictures.

2. **View Scalability**: The invention supports view scalability, enabling the adaptation of the number of views to be decoded and displayed based on network bandwidth and decoder capabilities. This is facilitated by the use of view scalability information (VSS) SEI messages, which indicate the dependencies between views.

3. **Parallel Processing**: The invention introduces a parallel decoding information SEI message to enable parallel processing of different views. This is crucial for real-time decoding in 3D TV applications, where multiple views need to be decoded simultaneously.

4. **Robustness**: The invention includes error resilience mechanisms to ensure the reliability of MVC bitstreams when transmitted over lossy channels. This is achieved through the use of error resilient tools and systematic restrictions on reference areas.

5. **Backward Compatibility**: The invention ensures backward compatibility with existing H.264/AVC decoders. The base view of an MVC bitstream is coded independently and can be decoded by a standard H.264/AVC decoder.

6. **Buffer Management**: The invention provides optimal buffer management techniques to minimize memory consumption at the decoder. This is achieved through the use of time-first coding and efficient reference picture management.

7. **Random Access and View Switching**: The invention supports random access and view switching, enabling smooth navigation and switching between different views. This is facilitated by the use of anchor pictures and view-switching points.

## BRIEF DESCRIPTION OF THE DRAWINGS

Figure 1 illustrates the end-to-end architecture of different 3D video applications, showing the flow from multiview video capture to rendering at the client.

Figure 2 shows a typical prediction structure of MVC, utilizing both inter-view prediction and hierarchical temporal scalability.

Figure 3 depicts examples of priority id assignments for NAL units in a 3-view bitstream with two levels of temporal resolution.

Figure 4 illustrates the view-first coding structure, where pictures of each view are contiguous in decoding order.

Figure 5 illustrates the time-first coding structure, where pictures of any temporal location are contiguous in decoding order.

Figure 6 shows the steps to reach the maximum decoded picture buffer (DPB) size for time-first coding.

Figure 7 illustrates the coding structure for a 3D TV system displaying two views simultaneously.

Figure 8 demonstrates the reference area restrictions for parallel decoding of macroblocks in view 1 and view 0.

Figure 9 illustrates the parallel decoding operation of two views using two processors.

## DETAILED DESCRIPTION OF VARIOUS EMBODIMENTS

### Efficient Compression

The invention utilizes inter-view prediction to exploit the correlation between views, thereby improving compression efficiency. In the MVC standard, decoded pictures of other views can be used as reference pictures when coding a picture, as long as they share the same capturing or output time. This is achieved through the use of inter-view prediction, where the view dependencies are defined for each coded video sequence. The prediction structure of MVC can be arranged flexibly to optimize compression efficiency.

### View Scalability

The invention supports view scalability, allowing the adaptation of the number of views to be decoded and displayed based on network bandwidth and decoder capabilities. This is facilitated by the use of view scalability information (VSS) SEI messages, which indicate the dependencies between views. The VSS SEI message provides a mapping between each operation point (identified by the combination of required view id values and temporal id values) and the required NAL units. This enables the server or media gateway to extract the required bitstream subset by discarding non-required NAL units.

### Parallel Processing

The invention introduces a parallel decoding information SEI message to enable parallel processing of different views. This is crucial for real-time decoding in 3D TV applications, where multiple views need to be decoded simultaneously. The parallel decoding information SEI message indicates that the views are encoded with systematic constraints, allowing any macroblock in a certain view to depend only on a subset of macroblocks in other views. This systematic restriction of reference areas enables parallel decoding of macroblocks in different views, significantly reducing the computational load.

### Robustness

The invention includes error resilience mechanisms to ensure the reliability of MVC bitstreams when transmitted over lossy channels. This is achieved through the use of error resilient tools and systematic restrictions on reference areas. The MVC standard provides error robustness by extending reference picture selection and redundant picture mechanisms to the view dimension. This strengthens the error resilience of the MVC bitstreams.

### Backward Compatibility

The invention ensures backward compatibility with existing H.264/AVC decoders. The base view of an MVC bitstream is coded independently and can be decoded by a standard H.264/AVC decoder. The base view is compliant with H.264/AVC, and the coded picture information for the base view is included in the VCL NAL units specified in H.264/AVC. A new NAL unit type, called coded slice of MVC extension, is used for containing coded picture information for non-base views. Prefix NAL units are introduced to indicate the essential characteristics of the base view-coded pictures in the multiview context.

### Buffer Management

The invention provides optimal buffer management techniques to minimize memory consumption at the decoder. This is achieved through the use of time-first coding and efficient reference picture management. In time-first coding, pictures of any temporal location are contiguous in decoding order, allowing the definition of access units that contain all the NAL units pertaining to a certain time instance. The optimal decoded picture buffer (DPB) size is determined by the highest temporal level of all the pictures and the number of views. The DPB management processes, including storage, marking, and output and removal of decoded pictures, are specified to efficiently utilize the buffer memory.

### Random Access and View Switching

The invention supports random access and view switching, enabling smooth navigation and switching between different views. Random access points are provided by instantaneous decoding refresh (IDR) pictures and anchor pictures. All random access points can also be used as view switching points. The invention also supports gradual decoding refresh (GDR) based on isolated regions technology, which can be indicated by the recovery point SEI message. View switching must occur at view-switching points, after which the new target view(s) can be correctly decoded.

### Additional Features

The invention includes additional features to enhance the overall performance and usability of the MVC system. These features include:

- **Reference Picture List Construction**: The reference picture list construction process can flexibly arrange temporal and view prediction references, providing potential coding efficiency gains and error resilience.
- **Active View Information SEI Message**: The active view information SEI message indicates the views that are to be output, allowing the decoder to decode and store only the required views.
- **Multiview Scene Information and Multiview Acquisition Information SEI Messages**: These SEI messages signal camera parameters, which are helpful in view interpolation by a renderer.

### Conclusion

The present invention provides a comprehensive solution for multiview video coding, addressing the challenges of efficient compression, view scalability, parallel processing, and robustness in transmission. The invention ensures backward compatibility with existing H.264/AVC decoders and provides optimal buffer management techniques to minimize memory consumption at the decoder. The invention is particularly useful for 3D video applications, such as free-viewpoint video, 3D TV, and immersive teleconferencing, where efficient and reliable multiview video coding is essential.