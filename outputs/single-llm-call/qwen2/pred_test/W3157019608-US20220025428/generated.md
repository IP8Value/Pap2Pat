# DESCRIPTION

## FIELD OF THE INVENTION

The present invention relates to the field of digital data storage and retrieval, specifically to a novel method and system for storing and retrieving digital information using nucleic acid-based memory (NAM). The invention leverages advancements in DNA nanotechnology and super-resolution microscopy to encode and read digital data stored in short oligonucleotide strands that are organized into DNA origami nanostructures.

## BACKGROUND OF THE INVENTION

As the global datasphere continues to grow, traditional memory materials are approaching their physical and economic limits. New non-volatile memory materials are needed to meet the increasing demand for data storage. DNA, with its high information density, significant retention time, and low energy of operation, is a viable alternative for data storage. However, existing methods for DNA-based data storage often rely on sequencing to recover the stored information, which can be time-consuming and resource-intensive.

The present invention, digital Nucleic Acid Memory (dNAM), addresses these limitations by using super-resolution microscopy techniques, specifically DNA-Points Accumulation for Imaging in Nanoscale Topography (DNA-PAINT), to read the stored data. In dNAM, digital information is encoded into specific combinations of single-stranded DNA (staple strands) that form DNA origami nanostructures. The staple strands are arranged at addressable locations within the origami, enabling site-specific localization of digital information. The presence or absence of specific staple strands defines binary states (1 or 0), which are read using DNA-PAINT.

## BRIEF SUMMARY OF THE INVENTION

The invention provides a method and system for digital Nucleic Acid Memory (dNAM) that uses DNA origami and super-resolution microscopy to store and retrieve digital information. The key aspects of the invention include:

1. **Encoding**: Digital information is encoded into specific combinations of single-stranded DNA (staple strands) that form DNA origami nanostructures. Each staple strand has two domains: one that forms a sequence-specific double helix with the scaffold strand, determining the address of the data within the origami, and another that extends above the origami and provides a docking site for fluorescently labeled imager strands.

2. **Origami Formation**: The staple strands are combined with a scaffold strand to form DNA origami nanostructures. The staple strands are arranged at addressable locations within the origami, defining an indexed matrix of digital information.

3. **Reading**: The DNA origami nanostructures are imaged using DNA-PAINT, a super-resolution microscopy technique. The presence or absence of the data domain is determined by monitoring the binding of fluorescently labeled imager strands, enabling the recovery of the encoded digital information.

4. **Error Correction**: The invention includes error-correcting algorithms that combine fountain codes with a custom, bi-level, parity-based, and orientation-invariant error detection scheme. These algorithms ensure the accurate recovery of the encoded message, even in the presence of errors.

## DETAILED DESCRIPTION

### Definitions

- **Digital Nucleic Acid Memory (dNAM)**: A method and system for storing and retrieving digital information using DNA origami and super-resolution microscopy.
- **Staple Strands**: Single-stranded DNA molecules that form specific structures when combined with a scaffold strand.
- **Scaffold Strand**: A long single-stranded DNA molecule that serves as a template for the formation of DNA origami nanostructures.
- **DNA-PAINT**: A super-resolution microscopy technique that uses transient binding of fluorescently labeled imager strands to visualize individual DNA molecules.
- **Fountain Codes**: A type of erasure code that allows the recovery of a message from a potentially infinite stream of encoded packets.
- **Parity Bits**: Bits used to check the integrity of data by ensuring that the total number of 1s in a data unit is even or odd.
- **Checksum**: A form of redundancy check used to detect errors in data.

### Nucleic Acid Architecture

The invention utilizes DNA origami to store digital information. DNA origami is a method of folding long single-stranded DNA (scaffold strand) into a desired shape using shorter single-stranded DNA (staple strands). In dNAM, the staple strands are designed to form a 6 × 8 matrix within the origami, with each site representing a binary state (1 or 0). The presence of a specific staple strand at a site indicates a binary 1, while the absence of the staple strand indicates a binary 0. The staple strands are further divided into two domains: the first domain forms a double helix with the scaffold strand, determining the address of the data within the origami, and the second domain extends above the origami and provides a docking site for fluorescently labeled imager strands.

### Dyes

Fluorescently labeled imager strands are used to read the digital information stored in the DNA origami. These imager strands bind transiently to the extended staple strands, emitting a signal that can be detected using DNA-PAINT. The binding and unbinding of the imager strands create a series of blinks, which are used to localize the position of the data domains within the origami. The imager strands are labeled with a variety of fluorophores, such as Cy3B, to enable multiplexing and increase the information density.

## EXAMPLES

### Example 1

#### Encoding a Message

To demonstrate the functionality of dNAM, the message "Data is in our DNA!" was encoded into 15 distinct DNA origami nanostructures. Each origami was designed with a 6 × 8 data matrix, with data domains positioned approximately 10 nm apart. The message was converted to binary code (ASCII) and segmented into 15 overlapping data droplets, each 16 bits long. Each origami was designed to contain a 4-bit binary index (0000–1110), 20 bits for parity checks, 4 bits for checksums, and 4 bits allocated as orientation markers.

#### Synthesis and Assembly

The staple strands and scaffold strand were mixed in a buffer containing 0.5× TBE and 18 mM MgCl2. The mixture was thermally cycled to fold the scaffold strand into the desired origami shape. The folded origami were purified by agarose gel electrophoresis and stored in the dark at 4°C.

#### Deposition and Imaging

The purified origami were deposited onto a glass coverslip using a glow discharge technique. The coverslip was then imaged using DNA-PAINT. A 561 nm laser source excited the fluorescently labeled imager strands, and the emitted fluorescence was captured using an EMCCD camera. The images were processed to identify the positions of the data domains and convert them to a 6 × 8 binary matrix.

#### Decoding the Message

The binary matrices were passed through a decoding algorithm that used a combination of fountain codes and bi-level parity checks to correct errors and recover the original message. Despite the presence of errors in some of the origami, the decoding algorithm successfully recovered the entire message "Data is in our DNA!" from a single super-resolution recording.

## MATERIALS AND METHODS

### Buffers

Two buffers were used to prepare and image DNA origami: a deposition buffer and an imaging buffer. The deposition buffer contained 0.5× TBE and 18 mM MgCl2. The imaging buffer contained the deposition buffer supplemented with 60 nM PCD, 1 mM Trolox, 3 nM imager strands, and 10 mM PCA. PCA was added to the imaging buffer immediately before the start of a DNA-PAINT recording.

### Encoding Algorithm

The encoding algorithm used a multi-layer error correction scheme to encode message data bits along with index, orientation, and error correction bits onto multiple origami. The algorithm first divided the message into segments and used a fountain code to generate data droplets. Each droplet was then encoded onto a 6 × 8 matrix, with the addition of index, orientation, parity, and checksum bits. The layout of the data, orientation, and index bits relative to the corresponding parity and checksum bits was invariant to rotation, making it possible for the error correction algorithm to perform error detection and recovery before determining the orientation.

### DNA Origami Folding

Rectangular DNA origami structures (~90 × 70 nm) were designed with 48 potential docking strand sites arranged in a 6 × 8 matrix with 10 nm spacing. The staple strands were selected to fold the M13 scaffold into the designed shape, with extended strands located at the '1' positions described in the design matrix. The origami were assembled by combining 22 nM M13mp18 with 10× unmodified strands, 50× extended strands, 1× TAE, and 18 mM MgCl2. The mixture was thermally cycled to fold the scaffold strand into the desired shape and then purified by agarose gel electrophoresis.

### Glass Coverslip Preparation

Borosilicate glass coverslips were sonicated in 0.1% (v/v) Liquinox and nano-pure water to remove contaminants and dried at 40°C for at least 30 minutes. Fiducial markers (200 µL of 0.2 pM AuNPs) were deposited onto the coverslips for 10 minutes at room temperature. The labeled coverslips were rinsed with methanol and nano-pure water and stored at 40°C prior to use.

### DNA-PAINT Fluorophore Localization

After recording a DNA-PAINT stack, the center position of signals (localizations) emitted by imager probes, transiently binding to DNA origami docking strands, were identified using the ImageJ ThunderSTORM plugin. The localizations were rendered and drift corrected using the Picasso-Render software package. Data visualization and peak fitting of image data for PSF analysis were performed using OriginPro Version 2019b.

### Localization Data Processing

A custom algorithm was developed to identify clusters of localizations, determine the maximum likelihood position of the emitters, and generate binary matrix data. The algorithm selected localization clusters at random from the localization list, determined the average position of nearby localizations, and counted the localizations within a radius (R) and the localizations within a band R < r < 2R. The algorithm accepted clusters if the counts in the inner circle were greater than a threshold and the counts in the outer band were less than 15% of the counts in the inner band. The algorithm then fit the cluster localizations to a grid of emitters using a maximum likelihood estimation. Signals that did not align to the 6 × 8 grid were filtered to minimize fragmented origami and to reduce inadvertent assimilation of the triangular origami fiducial markers into the results.

### Decoding Algorithm

The decoding algorithm utilized a multi-layer error correction/encoding scheme to recover the data in the presence of errors. The algorithm first worked at the dNAM origami level, using the parity and checksum bits to identify and correct errors and recover the correct matrix. After recovery, the algorithm used binary operations to recover the original data segments from the droplets.

## DATA SIMULATION TEST

To test the robustness of the encoding and decoding algorithms, origami data were simulated with randomly generated messages and errors. Random binary messages of size m were created (for m = 160 to 12,800 bits, at 320-bit intervals). These messages were divided into segments, and droplets were formed using the fountain code algorithm and encoded onto origami, along with the corresponding index, orientation, and error-correcting bits. Ten in silico copies of each unique origami were created, and 0–9 bits flipped at random to introduce errors. The origami were decoded using the described algorithm, and the success rate of message recovery was evaluated.

### Code Availability

The code for the encoding and decoding algorithms, as well as the data processing and simulation scripts, is available upon request from the inventors.

### AUTHOR CONTRIBUTIONS

The invention was conceived and developed by a team of researchers. Specific contributions include:
- **Conceptualization and Design**: [Author 1]
- **Experimental Work**: [Author 2], [Author 3]
- **Data Analysis and Algorithm Development**: [Author 4]
- **Writing and Reviewing the Manuscript**: [Author 5]

### SUPPLEMENTAL MATERIALS AND METHODS

Additional details on the materials and methods used in the invention, including the sequences of the staple strands, the detailed protocol for DNA origami folding, and the parameters for DNA-PAINT imaging, are provided in the supplemental materials.

### SUPPLEMENTAL RESULTS

Supplemental results include detailed analyses of the error rates, the performance of the error correction algorithms, and the scalability of the dNAM system. These results provide further validation of the effectiveness and robustness of the dNAM approach.