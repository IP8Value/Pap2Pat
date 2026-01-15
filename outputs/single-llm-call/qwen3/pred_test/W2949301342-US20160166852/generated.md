# DESCRIPTION

## BACKGROUND OF INVENTION

- motivate optogenetics

Optogenetics has emerged as a transformative methodology for the precise, reversible, and cell-type-specific modulation of neuronal activity through the genetic expression of light-sensitive ion channels and pumps. Since its foundational development in the early 2000s, this technique has revolutionized neuroscience by enabling causal investigations into the functional roles of defined neural circuits in behavior, cognition, and disease states such as Parkinson’s disease, epilepsy, and major depressive disorder. The core principle relies on the targeted expression of microbial rhodopsins—such as channelrhodopsin-2, halorhodopsin, and archeorhodopsin—in specific neuronal populations, allowing these cells to respond to externally applied light with rapid changes in membrane potential. However, the clinical translation of this powerful tool has been severely constrained by the physical limitations of conventional light delivery systems. Current implementations require the surgical implantation of optical fibers to deliver visible-wavelength light directly to brain tissue, an invasive procedure that introduces tissue damage, inflammatory responses, and long-term scarring, all of which compromise the integrity of experimental outcomes and preclude widespread human application. Moreover, the penetration depth of visible and even near-infrared light in biological tissue is inherently limited due to strong scattering and absorption by endogenous chromophores such as hemoglobin and melanin. As a result, optogenetic stimulation remains largely confined to superficial cortical layers in small animal models, rendering it ineffective for probing deeper subcortical structures such as the thalamus, hypothalamus, or brainstem in larger mammals, including primates and humans. These constraints have created a critical unmet need for a non-invasive, deep-tissue optogenetic platform capable of achieving spatially precise neuromodulation without the requirement for implanted hardware or the risks associated with open-brain surgery. The field demands a paradigm shift that decouples light delivery from physical access, enabling control over neural activity at arbitrary depths within intact, living organisms. This invention addresses that need by introducing two novel, radiation-free and radiation-assisted modalities that fundamentally redefine the spatial and temporal boundaries of optogenetic intervention.

## BRIEF SUMMARY

- summarize optogenetics
- limitations of current optogenetics
- need for improvement
- introduce novel approach

Optogenetics enables the control of neuronal membrane potential through the expression of light-gated ion channels or pumps, allowing researchers to activate or silence specific neural populations with millisecond precision using visible light. Despite its extraordinary utility in preclinical research, current implementations are fundamentally limited by the poor tissue penetration of optical radiation and the necessity for invasive fiber-optic implants, which restrict its applicability to superficial brain regions and small animal models. The inability to non-invasively stimulate deep neural circuits has hindered the translation of optogenetics into clinical neurology and psychiatry. To overcome these barriers, this invention introduces two complementary, non-invasive approaches: X-ray optogenetics and ultrasound optogenetics. In X-ray optogenetics, ionizing X-rays are directed toward the brain, where they are absorbed by biocompatible, light-emitting nanoparticles—specifically nanophosphors—that convert the high-energy radiation into visible photons locally, thereby activating genetically encoded rhodopsins without the need for optical fibers. In ultrasound optogenetics, focused ultrasonic waves induce sonoluminescence in the presence of chemiluminescent agents, generating transient bursts of visible light within tissue that similarly trigger rhodopsin activation. Both methods eliminate the requirement for surgical implantation, enable deep-tissue penetration, and offer unprecedented spatial and temporal control over neural activity. These innovations represent a fundamental advancement in neuromodulation technology, bridging the gap between laboratory-scale optogenetics and viable clinical applications in humans.

## DETAILED DISCLOSURE

- introduce optogenetics
- describe rhodopsins
- explain optogenetic technique
- discuss limitations of lasers and LEDs
- introduce X-ray optogenetic technique
- describe X-ray optogenetic system
- explain light-emitting particles
- describe light-sensitive ion channels
- introduce ultrasound optogenetic technique
- describe ultrasound optogenetic system
- explain chemiluminescent agents
- describe light-sensitive ion channels
- discuss X-ray optogenetic systems and methods
- explain X-ray-excitable light-emitting particles
- describe nanophosphors
- discuss up-conversion nanoparticles
- discuss UV/X-ray excitable nanoparticles
- introduce Table 1
- discuss nanophosphor properties
- explain importance of solubility
- discuss conversion efficiency
- explain importance of size distribution
- discuss coating of nanoparticles
- explain importance of biocompatibility
- discuss placement of light sources
- explain functionalization of light-emitting particles
- introduce Table 2
- discuss light-sensitive ion channels/pumps
- explain targeting of ion channels/pumps
- discuss focusing element
- explain poly-capillary lens
- discuss advantages of focused X-rays
- describe X-ray excitable nanophosphors
- introduce inverse distance weighting approximation
- derive X-ray intensity distribution equation
- discuss Fresnel zone plate for X-ray focusing
- describe pulsing X-ray emission using carbon-nanotube field-emission cathode
- introduce ultrasound optogenetics
- explain sonoluminescence effect
- discuss chemiluminescent agent enhancement
- describe targeting chemiluminescent agent to rhodopsins
- illustrate sonoluminescence for ion channel stimulation
- discuss advantages of X-optogenetics and U-optogenetics
- describe functionalization of light-emitting particles
- discuss X-ray flux and light-emitting particle size distribution
- introduce carbon nanotube X-ray source for temporal control
- discuss targeting light-emitting particles to rhodopsins
- describe millisecond control over X-ray delivery
- discuss minimizing radiation dose
- introduce U-optogenetics via sonoluminescence
- discuss advantages of X-optogenetics and U-optogenetics over related art
- describe computer-readable media for storing code and data
- define computer-readable medium
- explain meaning of "about" in numerical values
- introduce exemplified embodiments of the subject invention

Optogenetics is a technique that enables the control of cellular activity through the expression of light-sensitive proteins derived from microorganisms, particularly microbial rhodopsins. These proteins, when expressed in neurons, function as either ion channels or ion pumps that alter membrane potential upon absorption of photons within the visible spectrum. Channelrhodopsin-2 (ChR2), for instance, permits the influx of cations upon exposure to blue light, leading to membrane depolarization and neuronal excitation, while halorhodopsin (NphR) and archeorhodopsin (Arch) mediate chloride or proton efflux, respectively, resulting in hyperpolarization and inhibition. Traditional optogenetic protocols rely on external light sources such as lasers or light-emitting diodes (LEDs) to deliver wavelengths between 450 and 600 nanometers to the target tissue. However, these sources suffer from critical limitations: lasers are expensive, require precise alignment, and are typically delivered via invasive optical fibers, while LEDs produce diffuse, non-collimated light that lacks the spatial resolution necessary for single-cell or small-population targeting. Furthermore, both light sources are attenuated exponentially within biological tissue, rendering them ineffective beyond a few millimeters of depth—insufficient for reaching deep brain structures in humans. To overcome these constraints, this invention introduces two novel modalities: X-ray optogenetics and ultrasound optogenetics. In X-ray optogenetics, a source of X-rays is directed toward the target tissue, where it interacts with biocompatible, X-ray-excitable nanophosphors that have been delivered to the vicinity of rhodopsin-expressing neurons. These nanophosphors absorb high-energy X-ray photons and re-emit lower-energy photons in the visible range, which then activate the genetically encoded rhodopsins. The nanophosphors may be composed of materials such as Gd₂O₂S or LiGa₅O₈ doped with europium or terbium ions, chosen for their high conversion efficiency and tunable emission spectra. The emission wavelength is selected to match the absorption peak of the target rhodopsin, ensuring efficient activation. In parallel, ultrasound optogenetics employs focused ultrasonic waves to induce sonoluminescence—a phenomenon wherein microbubbles in tissue collapse under acoustic pressure, generating transient flashes of light. When chemiluminescent agents such as fluoresceinyl Cypridina luminescent analog (FCLA) are present, these flashes are amplified and spectrally tuned to the optimal activation range for rhodopsins. The chemiluminescent agent is delivered to the target region and functionalized to localize near rhodopsin-expressing cells, ensuring that the emitted photons are in close proximity to their targets. Both systems benefit from the use of focusing elements such as polycapillary lenses or Fresnel zone plates, which collimate and concentrate the X-ray beam to a microscale spot, enabling precise spatial targeting of neuronal populations without irradiating surrounding tissue. The X-ray source may be implemented using a carbon-nanotube field-emission cathode, which allows for rapid, millisecond-scale pulsing of X-rays, matching the temporal requirements of optogenetic stimulation. The nanophosphors must be engineered with specific physicochemical properties: they must be soluble in physiological media, exhibit narrow size distribution (preferably under 100 nm for blood-brain barrier penetration), and be coated with biocompatible polymers such as polyethylene glycol to minimize immune recognition and enhance circulation. Functionalization of the nanophosphors with antibodies or peptides that bind specifically to rhodopsin epitopes ensures that light emission occurs within nanometers of the target protein, maximizing activation efficiency. Similarly, chemiluminescent agents may be conjugated to targeting ligands to achieve localized accumulation. The system may be controlled by computer-readable media storing executable instructions for coordinating X-ray or ultrasound parameters—such as intensity, duration, frequency, and spatial focus—with real-time feedback from imaging or electrophysiological recordings. The term “about” as used herein refers to a tolerance of ±10% around a stated numerical value, accounting for inherent variability in biological systems and device calibration. The invention encompasses a broad range of embodiments, each defined by specific combinations of components, including the type of rhodopsin, the nature of the light-emitting particle, the excitation modality, the targeting strategy, and the delivery system.

### Embodiment 1

- introduce optogenetics method using X-rays and light-emitting particles

An optogenetic method is disclosed wherein a subject is administered a population of light-emitting particles that are capable of converting X-ray radiation into visible photons. These particles are delivered to a target region of the nervous system, where they accumulate in proximity to neurons genetically modified to express a light-sensitive ion channel or pump. Upon exposure of the target region to a controlled beam of X-rays, the particles emit visible light locally, which then activates the expressed rhodopsin, resulting in a change in membrane potential of the neuron. The method does not require the surgical implantation of optical fibers or any other invasive light delivery apparatus, and the X-ray source is positioned externally relative to the subject’s body. The emitted visible light is generated only in the immediate vicinity of the particles, ensuring spatial specificity and minimizing off-target activation. The duration and intensity of the X-ray exposure are calibrated to produce a sufficient photon flux to activate the rhodopsin without inducing thermal or radiological damage to surrounding tissue.

### Embodiment 2

- specify light-emitting particles as nanoparticles

The method of Embodiment 1 is further characterized in that the light-emitting particles are nanoparticles, each having a diameter less than one micrometer. The nanoparticle size is selected to facilitate systemic delivery, including passive or active transport across the blood-brain barrier. The nanoparticles are synthesized to have a core-shell architecture, with a crystalline inorganic core capable of X-ray-to-visible photon conversion and a surface coating that enhances colloidal stability and biocompatibility. The nanoparticles are administered intravenously or via intracerebroventricular injection, depending on the target anatomical location.

### Embodiment 3

- specify light-emitting particles as nanophosphors

The method of Embodiment 2 is further characterized in that the nanoparticles are nanophosphors composed of inorganic luminescent materials such as Gd₂O₂S, LiGa₅O₈, or NaYF₄ doped with europium, terbium, or chromium ions. These materials exhibit high quantum efficiency under X-ray excitation and emit photons with wavelengths between 450 and 600 nanometers, which correspond to the activation spectra of channelrhodopsin-2, halorhodopsin, or archeorhodopsin. The nanophosphors are engineered to have a narrow emission bandwidth and a high conversion efficiency of at least 10,000 visible photons per MeV of absorbed X-ray energy.

### Embodiment 4

- specify light-emitting particles as visible-light-emitting particles

The method of Embodiment 3 is further characterized in that the nanophosphors emit light exclusively within the visible spectrum, with no significant emission in the ultraviolet or infrared ranges. This spectral specificity ensures that the emitted photons are optimally matched to the absorption maxima of the rhodopsins, minimizing energy waste and reducing the risk of unintended phototoxicity or activation of endogenous photoreceptors.

### Embodiment 5

- specify X-ray source as carbon nanotube X-ray source

The method of Embodiment 1 is further characterized in that the X-ray source is a carbon-nanotube field-emission cathode, capable of generating soft X-rays in the energy range of 10 to 60 keV. The cathode is configured to emit X-rays in pulsed mode with temporal resolution of less than one millisecond, enabling precise control over the timing and duration of neuronal activation. The source is capable of repeated pulsing without thermal degradation, allowing for sustained optogenetic stimulation protocols over extended periods.

### Embodiment 6

- describe carbon nanotube X-ray source

The carbon-nanotube X-ray source comprises an array of vertically aligned carbon nanotubes mounted on a conductive substrate, which, when subjected to a high electric field, emits electrons via field emission. These electrons are accelerated toward a metal anode, producing X-rays through bremsstrahlung and characteristic radiation. The source is compact, requires no filament heating, and can be switched on and off instantaneously, enabling millisecond-scale temporal control of X-ray emission. The emission spectrum is tunable by adjusting the accelerating voltage and anode material, and the source is compatible with focusing optics such as polycapillary lenses or Fresnel zone plates.

### Embodiment 7

- specify sample as animal brain

The method of Embodiment 1 is further characterized in that the target tissue is the brain of a non-human animal, selected from the group consisting of mouse, rat, rabbit, or non-human primate. The X-ray source is positioned externally to the skull, and the light-emitting particles are delivered via systemic or local injection. The method enables non-invasive neuromodulation of deep brain structures such as the hippocampus, substantia nigra, or amygdala without the need for craniotomy or fiber implantation.

### Embodiment 8

- specify sample as mammal brain

The method of Embodiment 7 is further characterized in that the mammal is a large mammal, including but not limited to pig, sheep, or non-human primate. The X-ray flux and particle concentration are adjusted according to the increased tissue thickness and scattering properties of the larger brain, and the focusing element is calibrated to achieve a focal spot size of less than 500 micrometers at the target depth.

### Embodiment 9

- specify sample as human brain
- introduce focusing element for X-rays
- specify focusing element as lens
- specify lens as poly-capillary lens
- specify focusing element as zone plate
- specify zone plate as Fresnel zone plate
- specify focusing element as grating
- introduce X-ray stop or detector
- introduce rhodopsins
- transfect neurons with DNA encoding for rhodopsin
- functionalize light-emitting particles to bind to rhodopsins
- specify rhodopsin as channelrhodopsin2 (ChR2)
- specify rhodopsin as halorhodpsin (NphR)
- specify rhodopsin as archeorhodopsin (Arch)
- describe changing membrane potential of neuron
- introduce chemiluminescent agents
- specify chemiluminescent agents as FCLA molecules
- specify chemiluminescent agents as emitting light in visible spectrum
- introduce ultrasonic waves
- specify ultrasonic waves frequency as kHz or MHz
- specify sample as animal brain
- specify sample as mammal brain
- specify sample as human brain

The method of Embodiment 1 is further characterized in that the subject is a human, and the target tissue is the human brain. Neurons within the target region are genetically transfected with a viral vector encoding a light-sensitive ion channel or pump selected from the group consisting of channelrhodopsin-2 (ChR2), halorhodopsin (NphR), and archeorhodopsin (Arch). The light-emitting particles are functionalized with monoclonal antibodies or peptide ligands that bind specifically to extracellular epitopes of the expressed rhodopsin, ensuring subcellular localization. The X-ray source is coupled with a focusing element selected from the group consisting of a polycapillary lens, a Fresnel zone plate, or a diffraction grating, which collimates the X-ray beam to a focal spot of less than one millimeter in diameter at the target depth. A radiation detector or X-ray stop is positioned on the opposite side of the head to monitor beam transmission and ensure dose uniformity. In an alternative embodiment, the method employs ultrasound optogenetics, wherein ultrasonic waves with a frequency between 100 kHz and 10 MHz are directed toward the brain, inducing sonoluminescence in the presence of chemiluminescent agents such as FCLA molecules, which emit visible light at a peak wavelength of approximately 532 nanometers. The FCLA molecules are conjugated to targeting ligands and delivered to the same region as the rhodopsin-expressing neurons. Upon ultrasonic activation, the emitted photons trigger isomerization of retinal within the rhodopsin, resulting in a change in membrane potential of the neuron, either depolarizing or hyperpolarizing the cell depending on the type of rhodopsin expressed. The method is applicable to animal, mammalian, and human subjects, and enables non-invasive, deep-tissue neuromodulation without ionizing radiation in the ultrasound variant.

## Example 1

- introduce X-optogenetics radiation dose
- summarize effective radiation dose values
- motivate micro/nanoscopic scale optogenetics
- specify NP requirements for X-optogenetics
- calculate power emitted from nanophosphors
- convert radiation dose to photons per microgram tissue
- approximate photons per NP
- analyze number of photons needed to activate rhodopsins
- discuss modifications to enhance photon emission
- disclaim modifications and incorporate references

A feasibility analysis of X-ray optogenetics demonstrates that an effective radiation dose of up to 10 millisieverts, equivalent to a single diagnostic CT scan, is sufficient to activate rhodopsin-expressing neurons when paired with optimized nanophosphors. Assuming a nanophosphor composition of Gd₂O₂S doped with europium, with a conversion efficiency of 60,000 visible photons per MeV of absorbed X-ray energy, and a particle diameter of 150 nanometers, the emitted photon flux per nanoparticle exceeds ten photons under the maximum permissible dose. Given that each rhodopsin molecule requires between one and three photons to undergo retinal isomerization and activate, this photon output is sufficient to trigger measurable membrane potential changes in targeted neurons. Enhancements in quantum efficiency through improved doping strategies, increased particle size, or hybrid core-shell architectures can further elevate photon yield without exceeding radiological safety thresholds. These modifications are consistent with established nanomaterial design principles and are supported by prior art in X-ray luminescence imaging. The invention does not claim any specific chemical modification of the nanophosphors beyond what is disclosed herein, and all such modifications are considered within the scope of the invention as enabled by the teachings herein.