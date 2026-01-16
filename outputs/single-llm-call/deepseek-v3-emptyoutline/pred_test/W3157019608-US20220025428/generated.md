Here is the complete patent application following the provided outline and research paper content:

# DESCRIPTION  

## FIELD OF THE INVENTION  

The present invention relates generally to the field of nucleic acid-based data storage systems. More specifically, the invention provides systems and methods for encoding, storing, and retrieving digital information using nucleic acid nanostructures combined with super-resolution microscopy techniques. The invention particularly relates to digital Nucleic Acid Memory (dNAM) systems comprising DNA origami structures with spatially addressable data domains that encode binary information through the presence or absence of specific nucleic acid sequences, wherein said information is read optically using DNA-PAINT (Points Accumulation for Imaging in Nanoscale Topography) microscopy.  

## BACKGROUND OF THE INVENTION  

Traditional memory storage materials, including semiconductor-based and magnetic tape systems, are approaching their physical and economic limits. While DNA has emerged as a promising alternative due to its high information density, durability, and low energy requirements, existing DNA storage systems rely on sequencing technologies that impose limitations on data accessibility and editing. Current nucleic acid memory (NAM) systems encode information directly into DNA sequences, requiring sequencing for data retrieval, which introduces bottlenecks in read speed and limits practical applications.  

There exists an unmet need for DNA-based memory systems that decouple information storage from sequencing while maintaining high information density and enabling rapid, parallel data retrieval. The present invention addresses these limitations by providing a novel optical readout system that combines DNA nanotechnology with super-resolution microscopy to achieve spatially addressable data storage at nanoscale resolution.  

## BRIEF SUMMARY OF THE INVENTION  

The invention provides a digital Nucleic Acid Memory (dNAM) system comprising:  

1) Programmable nucleic acid nanostructures, particularly DNA origami, containing spatially arranged data domains at predetermined positions, wherein each data domain represents a binary digit (bit) through the presence ("1") or absence ("0") of an extendable nucleic acid sequence;  

2) An encoding system that converts digital information into patterns of data domains on said nanostructures using fountain codes combined with multi-layer error correction schemes;  

3) A storage medium comprising said nanostructures either in solution or deposited on solid substrates;  

4) A readout system employing DNA-PAINT super-resolution microscopy to optically detect the presence or absence of data domains through transient binding of fluorescently labeled imager strands; and  

5) A decoding algorithm that reconstructs the original digital information from super-resolution microscopy data while compensating for errors through parity checks, checksums, and fountain code redundancy.  

Key advantages of the invention include:  
- Decoupling of data storage density from readout density by separating 3D storage from 2D optical reading  
- Massive parallelism enabled by simultaneous optical detection of thousands of nanostructures  
- Inherent error correction through redundant encoding schemes  
- Non-destructive data reading through reversible hybridization of imager strands  
- Programmable data editing through selective strand replacement  

## DETAILED DESCRIPTION  

### Definitions  

For purposes of this invention, the following terms shall have the meanings specified:  

"DNA origami" refers to nanostructures formed by the folding of a long single-stranded DNA scaffold (typically derived from M13 bacteriophage) through hybridization with multiple short staple strands, resulting in predetermined two- or three-dimensional shapes.  

"Data domain" refers to a specific spatial location on a DNA origami structure that encodes one bit of information through either: (a) the presence of an extended staple strand containing a docking sequence for fluorescent imager strands (representing "1"), or (b) the absence of such extension (representing "0").  

"DNA-PAINT" (Points Accumulation for Imaging in Nanoscale Topography) refers to a super-resolution microscopy technique that achieves nanometer-scale resolution through transient binding of fluorescently labeled imager strands to complementary docking sequences, generating stochastic blinking events that can be localized with high precision.  

"Fountain code" refers to an erasure code that divides data into multiple segments which are then combined in random combinations such that the original data can be recovered from any sufficiently large subset of the transmitted combinations.  

"Parity bit" refers to an additional bit added to a binary string that indicates whether the number of "1" bits is even or odd, used for error detection.  

"Checksum" refers to a small-sized datum derived from a block of digital data for the purpose of detecting errors that may have been introduced during storage or retrieval.  

### Nucleic Acid Architecture  

The nucleic acid memory system of the invention utilizes rectangular DNA origami structures approximately 90 × 70 nm in size, comprising a scaffold strand folded into shape by approximately 200 staple strands. Of these, 48 staple strands are designated as potential data domains arranged in a 6 × 8 matrix with 10 nm spacing. Each data domain comprises either:  

1) An extended staple strand containing:  
   - A first domain that binds the scaffold strand and determines the spatial position within the origami  
   - A second domain extending outward from the origami structure containing a docking sequence for fluorescent imager strands (representing binary "1"); or  

2) An unmodified staple strand that binds only to the scaffold strand without extension (representing binary "0").  

The specific pattern of extended and unmodified strands across the 48 data domains encodes digital information according to an encoding algorithm described below. Multiple origami structures collectively store a complete data file through fountain code encoding, where each origami contains a portion of the total data along with error correction bits.  

### Dyes  

The system employs fluorescent dyes attached to DNA imager strands that transiently bind to data domains representing "1". Preferred embodiments use Cy3B dye attached to a short oligonucleotide complementary to the docking sequence of extended staple strands. The dye exhibits blinking behavior when bound to the docking sequence, enabling super-resolution localization via DNA-PAINT microscopy. Alternative embodiments may employ multiple dyes with distinct emission spectra to enable multiplexed detection of additional information at each data domain.  

## EXAMPLES  

### Example 1  

As a proof of concept, the message "Data is in our DNA!\n" was encoded into 15 distinct DNA origami structures. The message was first converted to binary ASCII code (160 bits total) and divided into 10 segments of 16 bits each. Using a fountain code algorithm, these segments were combined via XOR operations to create 15 "droplets," each containing portions of the original data.  

Each droplet was encoded onto a 6 × 8 binary matrix representing one origami design, with additional bits allocated as follows:  
- 16 bits for the droplet data  
- 4 bits for indexing (identifying which droplet the origami contains)  
- 4 bits for orientation markers  
- 4 bits for checksums  
- 20 bits for parity checks  

The 15 origami designs were synthesized separately by combining:  
- M13mp18 scaffold strand (22 nM)  
- 10× excess of unmodified staple strands  
- 50× excess of extended staple strands at positions encoding "1"  
- 1× TAE buffer  
- 18 mM MgCl2  

The mixtures were thermally annealed from 90°C to 25°C over 12 hours, purified by agarose gel electrophoresis, and stored at 4°C. Atomic force microscopy confirmed proper folding of all 15 designs with data domains in correct positions.  

For reading, approximately 20 fmoles of mixed origami were deposited onto glow-discharge-treated glass coverslips and imaged via DNA-PAINT using:  
- 561 nm laser excitation  
- 300 ms exposure time  
- 40,000 frames per recording  
- Imaging buffer containing 3 nM Cy3B-labeled imager strands  

Super-resolution images were reconstructed from blinking events, with individual data domains localized to <10 nm precision. A custom algorithm converted localization patterns into binary matrices and applied error correction to successfully recover the original message from a single recording containing approximately 750 origami structures.  

## MATERIALS AND METHODS  

### Buffers  

Two primary buffers were used:  
1) Deposition buffer: 0.5× TBE buffer with 18 mM MgCl2 for origami storage and deposition  
2) Imaging buffer: Deposition buffer supplemented with:  
   - 60 nM protocatechuate 3,4-dioxygenase (PCD)  
   - 1 mM Trolox  
   - 3 nM fluorescent imager strands  
   - 10 mM protocatechuic acid (PCA) added immediately before imaging  

### Encoding Algorithm  

The encoding process comprised:  
1) Message segmentation: Dividing the binary message into k equal-sized segments  
2) Fountain code application: Creating n droplets (n > k) by XOR combinations of random segment subsets following a Soliton distribution  
3) Matrix construction: For each droplet:  
   a) Arranging 16 data bits in a 6 × 8 matrix  
   b) Adding 4 index bits identifying the droplet  
   c) Adding 4 orientation marker bits  
   d) Computing and adding 4 checksum bits from the data  
   e) Computing and adding 20 parity bits from bi-level parity checks  
4) Origami design: Converting each matrix into staple strand specifications, with "1" positions receiving extended strands containing imager docking sequences  

### DNA Origami Folding  

Origami were folded by:  
1) Mixing scaffold and staple strands in appropriate ratios  
2) Thermal annealing using a programmed temperature ramp from 90°C to 25°C over 12 hours  
3) Purification via agarose gel electrophoresis  
4) Extraction from excised gel bands  

### Glass Coverslip Preparation  

Coverslips were prepared by:  
1) Sonicating in detergent and water  
2) Drying at 40°C  
3) Depositing 150 nm gold nanoparticles as fiducial markers  
4) Glow discharge treatment immediately before origami deposition  

### Fluorescence Microscopy  

DNA-PAINT imaging was performed using:  
1) Inverted microscope with TIRF illumination  
2) 561 nm laser excitation  
3) 100× 1.49 NA oil immersion objective  
4) EMCCD camera acquiring 512 × 512 pixel images at 300 ms exposure  

### DNA-PAINT Fluorophore Localization  

Blinking events were localized using:  
1) ThunderSTORM plugin for ImageJ for initial localization  
2) Picasso-Render software for drift correction and rendering  
3) Custom clustering algorithm to identify origami structures  

### Localization Data Processing  

A maximum likelihood algorithm:  
1) Identified localization clusters corresponding to individual origami  
2) Fitted localizations to a 6 × 8 grid template  
3) Assigned binary values to each grid position based on localization density  
4) Generated binary matrices for decoding  

### Decoding Algorithm  

The two-stage algorithm:  
1) Corrected errors in individual origami data by:  
   a) Calculating parity and checksum mismatches  
   b) Assigning error probabilities to each bit  
   c) Performing greedy search to find most probable error-free matrix  
2) Reconstructed original message from multiple origami by:  
   a) Maintaining priority queue of droplets by complexity  
   b) Solving segments through XOR operations  
   c) Reassembling complete message when sufficient segments recovered  

## DATA SIMULATION TEST  

Simulations evaluated system performance by:  
1) Generating random messages (160-12,800 bits)  
2) Encoding via the described algorithms  
3) Introducing random errors (0-9 bit flips per origami)  
4) Measuring recovery rates  

Results demonstrated:  
- Linear scaling of origami requirements with message size up to ~5 kB  
- Robust recovery with <7.4 errors/origami (97.5% success)  
- Graceful degradation at higher error rates  

## CODE AVAILABILITY  

Custom software for encoding, localization processing, and decoding has been developed and may be licensed separately.  

## AUTHOR CONTRIBUTIONS  

The inventors conceived the dNAM concept, developed the encoding/decoding algorithms, designed and synthesized DNA origami, performed microscopy experiments, analyzed data, and wrote supporting software.  

## SUPPLEMENTAL MATERIALS AND METHODS  

Additional experimental details include:  
- Complete staple strand sequences  
- Alternative buffer formulations  
- Microscope calibration protocols  
- Software parameter optimizations  

## SUPPLEMENTAL RESULTS  

Extended characterization data show:  
- AFM validation of all 15 origami designs  
- Error rate distributions across experiments  
- Comparison of different fiducial markers  
- Multiplexing feasibility studies  

This completes the comprehensive patent application for the digital Nucleic Acid Memory system. The invention represents a significant advance in DNA-based information storage by enabling optical reading of spatially encoded data with high density and robust error correction.