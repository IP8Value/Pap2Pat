# DESCRIPTION

## BACKGROUND

Visual hull techniques are widely used in image-based rendering and modeling of real objects. These methods leverage the concept of "Shape-from-Silhouette," which involves capturing multiple images of a target object from different viewpoints and constructing an approximate convex hull that contains the object. Compared to traditional geometry-based methods, visual hull techniques offer lower costs in data acquisition and produce more realistic results. However, these methods face significant challenges in accurately reconstructing the bottom of the object and concave shapes on the object's surface. The bottom of the object is often difficult to capture simultaneously with other surrounding reference images, leading to inaccuracies in the reconstructed model. Additionally, concave shapes cannot be distinguished from the object's silhouettes, making it challenging to accurately model these regions.

To address these issues, various research efforts have explored the use of planar mirrors and other auxiliary devices to enhance the visual hull reconstruction process. For example, some studies have utilized a single camera and two mirrors to generate stereo images, while others have employed a two-mirror system to capture multiple views of an object in a single image. However, these methods still struggle with modeling the bottom and concave surfaces accurately. The present invention introduces an improved visual hull rendering method that effectively solves these problems by utilizing a novel image acquisition platform involving a planar glass and a planar mirror. This platform allows for the simultaneous acquisition of images from both the top and bottom of the object, as well as from concave regions, without the need for complex alignment calculations. The resulting method produces high-quality, photorealistic models with accurate bottom and concave surface reconstructions.

## SUMMARY

The present invention provides an improved method for visual hull rendering that addresses the challenges of reconstructing the bottom and concave surfaces of an object. The method utilizes a novel image acquisition platform consisting of a planar glass and a planar mirror. This platform enables the simultaneous acquisition of images from the top and bottom of the object, as well as from concave regions, without the need for additional alignment calculations. The acquired images are then processed using an enhanced Image-Based Visual Hull (IBVH) algorithm to produce a more accurate and photorealistic 3D model of the object.

The key features of the invention include:
1. **Simultaneous Image Acquisition**: The platform captures images of the object from both the top and bottom sides, as well as from concave regions, in a single image. This eliminates the need for complex alignment calculations and ensures that all images are in the same reference frame.
2. **Virtual Camera and Virtual Image Concept**: The method introduces the concept of virtual cameras and virtual images to handle the bottom and concave surface rendering. Virtual cameras are created by mirroring the actual camera positions and orientations, allowing the bottom and concave images to be treated as additional reference images in the IBVH framework.
3. **Efficient Calibration**: The platform uses a simple camera calibration system involving colored concentric circles on the mirror to compute the camera parameters. This ensures that the images are accurately registered and ready for processing.
4. **Bottom Rendering Algorithm**: The method includes a specialized algorithm for rendering the object's bottom. This algorithm calculates the height of the bottom plane and adjusts the rendering process to ensure that the bottom is correctly modeled, even when the camera is positioned below the object.
5. **Concave Surface Approximation**: The method also introduces a technique for approximating concave surfaces. By placing a "negative" silhouette cone inside the concave region, the algorithm improves the accuracy of the concave surface reconstruction and texture retrieval.

The invention significantly enhances the quality and realism of 3D models generated using visual hull techniques, making it particularly useful for applications such as archaeology, artwork exhibition, and virtual reality.

## DETAILED DESCRIPTION

### New Reference Image Acquiring Platform

The core of the invention is a novel image acquisition platform designed to overcome the limitations of traditional visual hull methods. The platform consists of a planar glass board and a planar mirror. The object is placed on the glass, which is positioned directly above the mirror. This setup allows the reflection of the object's bottom to be captured in the same image as the top view, eliminating the need for additional alignment calculations.

#### Platform Setup

1. **Planar Glass and Mirror**: The object is placed on a planar glass board, which is positioned above a planar mirror. The glass and mirror are aligned such that the reflection of the object's bottom is visible in the mirror.
2. **Camera Positioning**: A camera is used to capture images of the object from various viewpoints. The camera is calibrated using a simple system involving colored concentric circles on the mirror. The camera parameters are computed using common P4P calibration methods.
3. **Image Acquisition**: The platform captures images of the object from both the top and bottom sides, as well as from concave regions, in a single image. This is achieved by leveraging the reflection properties of the mirror and the transparency of the glass.

### Virtual Camera and Virtual Image Concept

To handle the bottom and concave surface rendering, the invention introduces the concept of virtual cameras and virtual images. This concept allows the bottom and concave images to be treated as additional reference images in the IBVH framework.

#### Virtual Camera Creation

1. **Symmetrical Camera Positioning**: For each reference image, a virtual camera is created by mirroring the actual camera position and orientation about the mirror plane. The virtual camera is positioned at the symmetrical point of the actual camera, with the same orientation but mirrored along the X and Z axes.
2. **Parameter Calculation**: The parameters of the virtual camera are derived from the parameters of the actual camera. The internal parameters remain the same, while the translation and rotation matrices are adjusted to reflect the mirrored position and orientation.

#### Virtual Image Generation

1. **Image Segmentation**: Each reference image is segmented into two parts: the top image and the bottom image. The top image is used directly in the IBVH framework, while the bottom image is transformed into a virtual image.
2. **Virtual Image Transformation**: The virtual image is generated by flipping the bottom image vertically. This virtual image is then treated as an additional reference image in the IBVH framework.

### Bottom Rendering Algorithm

The bottom rendering algorithm ensures that the bottom of the object is correctly modeled, even when the camera is positioned below the object. The algorithm involves the following steps:

1. **Height Calculation**: The height of the bottom plane is calculated by marking several points at the edge of the object's bottom in the reference images. The corresponding 3D points on the visual hull are found, and their average Y-coordinate is used as an approximation of the bottom height.
2. **Viewpoint Adjustment**: When rendering the object from a new viewpoint, the algorithm adjusts the rendering process based on the height of the viewpoint:
   - If the viewpoint is higher than the bottom plane, the rendering process is the same as the regular IBVH method.
   - If the viewpoint is lower than the bottom plane, the algorithm projects the viewing ray through each pixel and calculates the intersection point with the bottom plane. The texture for the pixel is then retrieved based on this intersection point.

### Concave Surface Approximation

The concave surface approximation technique improves the accuracy of the concave surface reconstruction and texture retrieval. The technique involves the following steps:

1. **Concave Silhouette Cone**: A special image is taken from right above the concave region, and the silhouette of the concave region is extracted. A virtual camera is placed inside the concave region, and the silhouette is projected onto the image plane of the virtual camera to generate a virtual "negative" silhouette cone.
2. **Negative Cone Subtraction**: The virtual "negative" silhouette cone is subtracted from the intersection of other silhouette cones to approximate the concave surface. This process helps to correct the geometry and texture of the concave region.
3. **Concave Surface Rendering**: When rendering the object from a new viewpoint, the algorithm uses the virtual silhouette image to provide geometry information for the concave surface and the real images to provide texture information. The algorithm projects the viewing ray through each pixel and calculates the intersection points with the virtual and real silhouettes to determine the correct 3D correspondence and texture for the pixel.

### Experimental Results

The improved visual hull rendering method has been tested on several sets of input images. The results demonstrate that the method produces high-quality, photorealistic models with accurate bottom and concave surface reconstructions. The method significantly outperforms traditional IBVH methods in terms of both geometry and texture accuracy.

#### Example 1: Teapot Model

- **Input Images**: The input images show the teapot from various viewpoints, including the bottom and concave regions.
- **Rendered Views**: The rendered views are highly photorealistic, with the bottom of the teapot and the concave regions accurately reconstructed. The flat bottom is correctly modeled, and the texture is well-preserved.

#### Example 2: Cup Model

- **Input Images**: The input images show the cup from various viewpoints, including the bottom and concave regions.
- **Rendered Views**: The rendered views are highly photorealistic, with the bottom of the cup and the concave regions accurately reconstructed. The inner surface of the cup is correctly modeled, and the texture is well-preserved.

### Conclusion

The present invention provides a simple and efficient method for solving the bottom and concave surface rendering problems in image-based visual hull. The method utilizes a novel image acquisition platform involving a planar glass and a planar mirror to capture images of the object from both the top and bottom sides, as well as from concave regions, in a single image. The method introduces the concepts of virtual cameras and virtual images to handle the bottom and concave surface rendering, ensuring that the resulting 3D models are highly photorealistic and accurate. The method is particularly useful for applications such as archaeology, artwork exhibition, and virtual reality, where the shape and texture of the bottom and concave surfaces are critical. Future work will focus on improving the precision of the silhouette extraction and extending the method to handle more complex objects with multiple concave regions.