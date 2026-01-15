# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to plant hormone sensing, specifically to methods and devices for the non-invasive detection of methyl salicylate and related phenolic compounds in living plants as early indicators of pathogen infection. More particularly, the invention provides a hydrogel-based extraction system integrated with optical or electrochemical detection mechanisms that enable real-time, field-deployable monitoring of plant stress responses without damaging plant tissue. The invention further encompasses chemical receptors—such as diboroxane derivatives, phenylboronic acid derivatives, and boroxine derivatives—that selectively bind methyl salicylate and generate measurable fluorescence or electrochemical signals, thereby facilitating the early diagnosis of plant diseases before visible symptoms manifest.

## BACKGROUND ART

Plants possess sophisticated defense mechanisms to counteract biotic stresses such as bacterial, fungal, and viral infections. Upon pathogen recognition, plants activate systemic acquired resistance (SAR), a long-lasting immune response mediated by signaling molecules including salicylic acid and its volatile derivative, methyl salicylate. These phytohormones function not only locally at the infection site but also systemically to prime distal tissues for enhanced defense readiness. Concurrently, plants upregulate the biosynthesis of secondary metabolites, particularly phenylpropanoids like chlorogenic acid, scopoletin, and ferulic acid, which serve as phytoanticipins with antimicrobial properties. This metabolic reprogramming is often accompanied by the accumulation of blue-fluorescent phenolic compounds, detectable under ultraviolet excitation.

Plant hormone synthesis occurs through tightly regulated enzymatic pathways. Salicylic acid is primarily synthesized via the isochorismate pathway in chloroplasts, and it can be methylated by salicylic acid carboxyl methyltransferase (SAMT) to form methyl salicylate, a volatile signal that facilitates inter-plant communication and systemic defense activation. The signaling cascade initiated by these hormones involves receptor-mediated perception, transcriptional reprogramming, and the reinforcement of physical and chemical barriers against pathogens.

Despite advances in understanding plant defense biochemistry, practical field-based detection of early infection remains challenging. Conventional diagnostic methods such as enzyme-linked immunosorbent assays (ELISA) and polymerase chain reaction (PCR) require destructive sampling, laboratory infrastructure, and significant time, rendering them unsuitable for real-time agricultural monitoring. Image-based machine learning approaches have shown promise but suffer from limited generalizability due to variability in field conditions and insufficient training data. While intrinsic plant fluorescence has been explored for stress detection, chlorophyll-mediated quenching in green tissues severely compromises the accuracy of blue-green fluorescence measurements. Prior art in plant hormone sensing lacks non-invasive, selective, and quantitative tools capable of detecting low-abundance signaling molecules like methyl salicylate directly from intact leaves in situ.

## SUMMARY OF INVENTION

### Technical Problem

The technical problem addressed by the present invention is the absence of a practical, non-destructive, and sensitive method for the early detection of pathogen infection in crops through the direct measurement of plant-emitted defense-related volatile organic compounds, particularly methyl salicylate. Existing techniques either destroy plant tissue, require complex instrumentation, or fail to distinguish specific hormonal signals from background autofluorescence due to interference from chlorophyll and other pigments.

### Solution to Problem

The invention solves this problem by providing a hydrogel-based extraction platform combined with chemoselective boron-containing receptors that specifically bind methyl salicylate. The hydrogel, composed of agarose or similar biocompatible polymers, is applied directly to the leaf surface where it non-invasively extracts water-soluble phenolic compounds—including methyl salicylate—through natural cuticular defects without penetrating living cells. The embedded receptor molecules undergo a structural or electronic change upon binding methyl salicylate, resulting in either enhanced fluorescence emission or a measurable shift in electrochemical current. This signal is then quantified using portable optical or electrochemical readers, enabling rapid, on-site diagnosis of pre-symptomatic infection.

### Advantageous Effect of Invention

The invention offers several advantageous effects. First, it enables truly non-invasive monitoring, preserving plant integrity and allowing repeated measurements over time. Second, by extracting analytes into a hydrogel matrix that excludes chlorophyll and other interfering pigments, the method eliminates fluorescence quenching, significantly improving signal-to-noise ratio and quantification accuracy. Third, the use of boron-based receptors provides high selectivity for methyl salicylate over structurally similar compounds. Fourth, the system is low-cost, scalable, and operable by non-specialists in field conditions. Finally, early detection permits timely intervention—such as targeted pesticide application or removal of infected plants—thereby reducing crop losses and minimizing environmental impact.

## DESCRIPTION OF EMBODIMENTS

The invention introduces a novel receptor for methyl salicylate based on boron-oxygen chemistry. Specifically, the receptor comprises a diboroxane derivative, which features two boron atoms bridged by oxygen atoms in a cyclic structure, enhancing stability and binding affinity. Alternatively, a phenylboronic acid derivative may be employed, wherein a boronic acid group is attached to an aromatic ring, facilitating π–π stacking interactions with the phenolic moiety of methyl salicylate. A third embodiment utilizes a boroxine derivative, a trimeric cyclic structure formed from three boronic acid units, offering multivalent binding sites for improved sensitivity.

The reaction between the receptor and methyl salicylate involves the formation of a reversible covalent bond between the boron atom and the hydroxyl group of methyl salicylate, generating a boronate ester complex. This binding event alters the electronic environment of the receptor, leading to a pronounced fluorescence emission phenomenon. Under UV excitation at approximately 310 nm, the receptor-methyl salicylate complex emits blue fluorescence with a peak around 410–430 nm, depending on the specific receptor used. This emission is absent or minimal in the unbound state, providing a clear on/off signal.

In parallel, the electrochemical behavior of the receptor changes upon methyl salicylate binding. When immobilized on an electrode surface within an electrochemical cell, the receptor exhibits a measurable change in oxidation or reduction current. This change correlates linearly with methyl salicylate concentration, enabling quantitative analysis.

The method for detecting methyl salicylate involves placing a hydrogel film containing the receptor onto a plant leaf for a defined period (e.g., 1–3 hours). During this time, methyl salicylate and related phenolics diffuse from the apoplast into the hydrogel via leaching through cuticular micropores. The film is then irradiated with UV light, and emitted fluorescence is detected using a spectrometer or simple photodiode array. Alternatively, the hydrogel may be integrated with electrodes for direct electrochemical readout.

The methyl salicylate sensor comprises a recognition section—containing the boron-based receptor embedded in hydrogel—and a detection section configured for either optical or electrochemical signal transduction. Optical detection involves measuring fluorescence intensity at a characteristic wavelength and comparing it to a reference value derived from healthy plants. Electrochemical detection measures current changes in a three-electrode system (working, reference, counter) and compares the signal to baseline values.

A computer program may be integrated to automate signal analysis, applying calibration curves to convert raw signals into methyl salicylate concentrations. The output includes a diagnostic result indicating infection likelihood, concentration levels, and recommended actions. The sensor can be installed near crops—on stakes, drones, or robotic arms—for continuous or periodic monitoring. It is applicable to a wide range of crops, including tomatoes, tobacco, rice, and wheat, and detects diseases caused by pathogens such as *Ralstonia solanacearum*, *Pseudomonas syringae*, and *Fusarium* species.

### <Fluorescence Emission Phenomenon>

The fluorescence emission phenomenon arises from the photoexcitation of the receptor-methyl salicylate complex. Upon irradiation with excitation light at 310 nm from a xenon lamp or UV LED, electrons in the complex are promoted to a higher energy state. As they return to the ground state, they emit photons in the blue region (410–430 nm). This emission is highly specific to the bound state due to conformational rigidity induced by boronate ester formation, which reduces non-radiative decay. Detection is performed using a spectrofluorometer or compact fluorescence reader, with intensity measured at the emission peak. The value is compared to a reference threshold established from uninfected control plants; exceeding this threshold indicates pathogen-induced methyl salicylate accumulation.

### <Electrochemical Behavior>

The electrochemical behavior of the receptor is monitored in a standard electrochemical cell containing a working electrode coated with the receptor-hydrogel composite, a reference electrode (e.g., Ag/AgCl), and a counter electrode (e.g., platinum wire). Cyclic voltammetry or amperometry is used to measure current. Upon methyl salicylate binding, the electron-donating nature of the phenolic group alters the redox potential of the boron center, causing a detectable increase or decrease in current. The change in current value is proportional to analyte concentration. Comparison with a reference value—obtained from sensors placed on healthy plants—enables infection diagnosis. The electrochemical cell is miniaturized for field use, powered by batteries, and connected to a microcontroller for data logging.

### <Methyl Salicylate Sensor>

The methyl salicylate sensor integrates a recognition section comprising agarose hydrogel doped with a diboroxane, phenylboronic acid, or boroxine derivative receptor. The detection section supports either optical or electrochemical modalities. In optical mode, a UV light source excites the film, and a photodetector captures fluorescence, which is analyzed by onboard software to determine concentration. In electrochemical mode, embedded electrodes measure current changes. The computer program processes signals using preloaded calibration data, outputs infection risk levels, and may transmit alerts via wireless networks. The sensor measures methyl salicylate concentration quantitatively, enabling dose-response assessment of infection severity. Installed near crops—on greenhouse structures, field posts, or autonomous vehicles—it continuously monitors plant health. Validated crops include tomato, tobacco, potato, and cucumber; target diseases include bacterial wilt, blight, and rust.

### <Method for Early Detection of Pathogen Infection in Crop>

The method involves installing the methyl salicylate sensor on or near crop leaves. The hydrogel film contacts the leaf surface for 1–3 hours, extracting methyl salicylate released during early infection. Fluorescence or electrochemical signals are recorded and analyzed. A significant increase over baseline indicates pathogen presence before visual symptoms appear. Demonstrated on cherry tomato plants infected with *Ralstonia solanacearum*, the method detected elevated methyl salicylate as early as 3 days post-inoculation. Applicable crops include solanaceous species (tomato, pepper, eggplant), cereals (wheat, rice), and legumes. Diseases detectable include those caused by *Xanthomonas*, *Phytophthora*, and *Botrytis*.

## EXAMPLES

The following examples illustrate embodiments of the invention.

### Example 1

A filter paper impregnated with a triarylboroxine (TBO) receptor was placed on a cherry tomato leaf previously treated with salicylic acid via root uptake. After 3 hours, the paper exhibited strong blue fluorescence under 310 nm UV light, confirming extraction and binding.

### Comparative Example 1

A TBO-impregnated filter paper placed on a healthy, untreated leaf showed negligible fluorescence, demonstrating specificity.

### Example 2

TBO paper detected fluorescence when exposed to headspace volatiles from infected plants, proving capability to sense airborne methyl salicylate.

### Example 3

Phenylboronic acid (PB)-doped paper showed fluorescence upon contact with infected leaves.

### Comparative Example 2

PB paper on healthy leaves showed no signal.

### Example 4

Tris(4-fluorophenyl)boroxine (TFPB)-treated paper fluoresced with infected leaves.

### Comparative Example 3

TFPB on healthy leaves yielded no emission.

### Example 5

Tris(3,5-bis(trifluoromethyl)phenyl)boroxine (TBO)-functionalized paper detected methyl salicylate with high intensity.

### Comparative Example 4

TBO paper without methyl salicylate exposure showed baseline fluorescence. Fluorescence intensity of TBO increased linearly with methyl salicylate concentration (0.02–0.5 mmol/L, R²=0.999). Electrochemical measurements showed current increase with methyl salicylate but not with methyl jasmonate (MJA), confirming selectivity. The sensing method employs boron-oxygen compounds as receptors, with diboroxane, phenylboronic acid, and boroxine derivatives enabling fluorescence or electrochemical detection via current changes. The methyl salicylate sensor uses optical or electrochemical detection to identify pathogen infection. A dedicated program analyzes signals and outputs results. The invention is industrially applicable for precision agriculture.

## INDUSTRIAL APPLICABILITY

The invention is highly industrially applicable in modern agriculture, enabling smart farming through early disease detection. It can be integrated into IoT-based monitoring systems, drone surveillance, and automated greenhouse controls. By reducing reliance on broad-spectrum pesticides and enabling targeted interventions, it promotes sustainable crop protection, enhances yield stability, and lowers production costs. The low material cost and ease of use make it accessible to smallholder farmers globally.