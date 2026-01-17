# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to the field of medical imaging and, more specifically, to a non-invasive method for tracking tumors in image-guided radiation therapy (IGRT) systems. The invention provides a robust and accurate technique for estimating the position of tumors, particularly in the lung and abdominal areas, which are subject to significant motion due to respiration.

## BACKGROUND OF THE INVENTION

Tumor tracking is a critical component of image-guided radiation therapy (IGRT) systems, especially for treating tumors in the lung and abdominal areas. These tumors move significantly due to respiration, making it challenging to deliver precise radiation doses without affecting healthy tissue. Conventional tracking methods often rely on internal and external surrogates, such as implanted fiducials or external markers, to monitor tumor motion. However, these methods have several drawbacks, including invasiveness, potential harm to healthy tissues, and inaccuracies due to the complex nature of respiratory biomechanics.

Internal fiducials, such as metallic markers, require surgical implantation, which can cause complications like pneumothorax. Moreover, these fiducials can shift or relocate during multiple treatment sessions, leading to uncertainties in their reference positions. External markers, such as chest and abdominal pointers, can be used to estimate tumor position indirectly, but the correlation between external markers and tumor motion is often violated due to the complexity of respiratory movements.

Alternative methods, such as parametric models of motion patterns, require extensive manual labeling and training, which is time-consuming and patient-specific. Template matching methods, while effective in some cases, can fail in low-contrast regions where the image quality is poor.

To address these limitations, the present invention introduces a novel tumor tracking method that does not require any invasive internal fiducials or external markers. The method treats tumor tracking as a regression model fitting task between orthogonal X-ray videos and the underlying tumor motion. By leveraging image features and affine motion models, the invention provides a fast, accurate, and non-invasive solution for tumor tracking in IGRT systems.

## SUMMARY OF THE INVENTION

The present invention provides a method for non-invasively tracking the position of tumors in image-guided radiation therapy (IGRT) systems. The method involves learning an online regression model that correlates image features extracted from orthogonal X-ray videos to the underlying tumor motion. The regression model is trained using an initial pair of orthogonal X-ray images and can be applied to subsequent images to estimate the tumor position in real-time.

Key aspects of the invention include:
1. **Non-Invasive Tracking**: The method does not require the implantation of internal fiducials or the use of external markers, reducing the risk of complications and improving patient comfort.
2. **Regression Model Learning**: An online regression model is learned to map image features to tumor motion parameters. The model is trained using a set of randomly generated 3D motion hypotheses and corresponding feature vectors.
3. **Feature Extraction**: Image features, such as Histograms of Oriented Gradients (HOG), are extracted from the tumor region in the X-ray images. These features are sensitive to motion and provide a robust indicator of tumor movement.
4. **Geodesic Distance**: The regression model is optimized using geodesic distances on a Riemannian manifold, ensuring accurate and reliable motion estimation.
5. **Real-Time Performance**: The method is computationally efficient, allowing for real-time tumor tracking with minimal latency.
6. **Adaptive Window Selection**: The method includes an adaptive mechanism for selecting the optimal tracking window size, which enhances the accuracy and robustness of the tracking process.

The invention is particularly useful for tracking low-contrast tumors in the lung and abdominal areas, where traditional methods often fail. By providing accurate and continuous tumor tracking, the invention significantly reduces the need for large treatment margins, thereby minimizing the adverse effects of radiation on healthy tissue.

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS

### Effect of the Invention

The present invention offers a significant improvement in the accuracy and efficiency of tumor tracking in image-guided radiation therapy (IGRT) systems. By eliminating the need for invasive internal fiducials or external markers, the invention reduces the risk of complications and enhances patient comfort. The non-invasive nature of the method also allows for broader applicability across various types of tumors and patients.

The regression model fitting approach used in the invention provides a robust and accurate estimation of tumor motion, even in low-contrast regions. The use of geodesic distances on a Riemannian manifold ensures that the motion parameters are accurately represented, leading to more reliable tracking results. The computational efficiency of the method enables real-time tumor tracking, making it suitable for clinical settings where rapid and precise adjustments are required.

Furthermore, the adaptive window selection mechanism optimizes the tracking window size, enhancing the method's performance in different patient datasets. This adaptability ensures that the invention can handle a wide range of tumor locations and characteristics, making it a versatile solution for IGRT systems.

In summary, the present invention provides a non-invasive, accurate, and efficient method for tumor tracking in IGRT systems, significantly improving the precision and safety of radiation therapy.