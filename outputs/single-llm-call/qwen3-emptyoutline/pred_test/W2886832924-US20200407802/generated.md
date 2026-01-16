# DESCRIPTION

## FEDERAL FUNDING ACKNOWLEDGEMENT

This work was supported by grants from the National Institutes of Health, National Cancer Institute, including U24 CA143882 (PWL, BPB, HQD, and HS); R01 CA170550 (PWL); U01 CA184826 (BPB); U24 CA210969 (PWL, BPB, and HS); and by the National Institutes of Health, National Human Genome Research Institute grant R01 HG006705 (BPB). The content of this application is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health. The funding agencies played no role in the design of the invention, the interpretation of results, or the preparation of this disclosure. The invention described herein was made with government support under the aforementioned grants, and the government may have certain rights in this invention pursuant to 35 U.S.C. § 202 and applicable regulations.

## FIELD OF THE INVENTION

The present invention relates to methods and systems for identifying, quantifying, and interpreting patterns of DNA methylation in human and mammalian genomes, particularly those associated with cellular proliferation, developmental lineage, chronological age, and neoplastic transformation. More specifically, the invention provides novel diagnostic, prognostic, and therapeutic tools grounded in the discovery that a specific subset of cytosine-phosphate-guanine dinucleotides—those occurring as solo CpGs flanked on both sides by adenine or thymine nucleotides (WCGW motifs)—exhibit a unique and highly consistent pattern of hypomethylation across diverse cell types, tissues, and species. This hypomethylation signature is not random but is systematically linked to the replication timing of genomic domains, the absence of specific histone modifications, and the cumulative number of cell divisions experienced by a cell lineage since embryogenesis. The invention enables the precise mapping of these hypomethylated regions, their classification into stable, cell-type-invariant domains known as proliferative methylation domains (PMDs), and the use of these domains as quantitative biomarkers for biological age, mitotic history, cancer risk, tumor aggressiveness, and response to therapy. The invention further provides computational algorithms, sequencing-based assays, and analytical platforms that leverage this signature to detect and monitor disease states with unprecedented sensitivity, even in low-coverage or single-cell methylome data.

## BACKGROUND

DNA methylation is a fundamental epigenetic modification that plays a critical role in regulating gene expression, maintaining genomic stability, and controlling cellular identity. In mammalian genomes, methylation predominantly occurs at cytosine residues preceding guanine nucleotides (CpG dinucleotides), and the distribution of methylated and unmethylated CpGs is not uniform across the genome. Historically, CpG islands—regions of high CpG density often located near gene promoters—have been the primary focus of methylation studies due to their association with transcriptional silencing when hypermethylated. In contrast, large genomic regions exhibiting widespread hypomethylation, termed proliferative methylation domains (PMDs), have received less attention despite their prevalence in somatic tissues and their strong association with cancer. Early studies identified PMDs as megabase-scale domains of reduced methylation in cultured cells and tumors, but their origin, functional significance, and consistency across cell types remained poorly understood. 

Prior methods for identifying PMDs relied heavily on high-coverage whole-genome bisulfite sequencing (WGBS) and complex computational models, such as hidden Markov models, which required substantial sequencing depth and were sensitive to noise, sample heterogeneity, and technical artifacts. These approaches often failed to distinguish between true biological hypomethylation and artifacts arising from low cellularity, contamination by non-target cells, or incomplete bisulfite conversion. Moreover, the variability of PMD extent across samples—particularly between tumors and their adjacent normal tissues—led to conflicting interpretations regarding whether PMDs were cancer-specific phenomena or inherent features of somatic cell biology. 

A critical gap in the field was the inability to reliably detect PMDs in low-coverage sequencing data, limiting their utility in clinical settings where sample quantity is limited, such as in liquid biopsies, single-cell analyses, or archived tissue samples. Additionally, while some studies suggested that PMD hypomethylation correlated with aging or cell proliferation, no consistent molecular signature had been identified that could serve as a universal, quantitative, and reproducible metric for mitotic history across diverse tissues and species. The prevailing models attributed methylation loss to passive dilution during DNA replication, but lacked a mechanistic framework that could explain why certain CpG sites were consistently more vulnerable than others, or how this vulnerability was coordinated with nuclear architecture, replication timing, and chromatin state. 

Furthermore, prior attempts to link DNA methylation to chronological age relied on CpG sites located in promoter regions or CpG islands, which are subject to dynamic regulation by transcriptional activity, environmental exposures, and tissue-specific differentiation programs. These age-associated methylation signatures, while useful, were confounded by cell-type composition, tissue heterogeneity, and non-mitotic influences such as inflammation or oxidative stress. As a result, there was no method available that could directly and quantitatively measure the cumulative number of cell divisions a cell lineage had undergone since embryonic development—a fundamental biological parameter that underlies aging, tissue regeneration, and oncogenesis. 

The present invention addresses these longstanding limitations by identifying a novel, sequence-defined subset of CpG dinucleotides—solo-WCGW CpGs—that exhibit a remarkably consistent and quantifiable pattern of hypomethylation across all somatic cell types, from embryonic tissues to aged adults, and from healthy individuals to patients with cancer. This discovery reveals that PMD structure is not a stochastic or cancer-specific phenomenon, but rather a conserved, developmentally programmed feature of the genome that reflects the mitotic history of cells. The invention leverages this signature to provide a new class of biomarkers that are independent of cell type, tissue origin, or sequencing depth, enabling unprecedented precision in the detection of biological age, cancer progression, and cellular proliferation dynamics.

## SUMMARY OF THE INVENTION

The present invention provides a novel method for identifying, quantifying, and utilizing a specific class of CpG dinucleotides—solo-WCGW CpGs—as a universal biomarker for cellular proliferation, developmental lineage, chronological age, and neoplastic transformation. A solo-WCGW CpG is defined as a cytosine-phosphate-guanine dinucleotide that is not adjacent to any other CpG within a window of thirty-five base pairs on either side and is flanked on both the 5’ and 3’ sides by adenine or thymine nucleotides, forming the tetranucleotide context WCGW. These CpGs are uniquely susceptible to hypomethylation across all somatic cell types and species, and their methylation status serves as a highly accurate and reproducible readout of the cumulative number of cell divisions experienced by a cell lineage since embryogenesis. 

The invention provides a system for determining the methylation level of solo-WCGW CpGs in a biological sample using high-throughput sequencing or microarray-based technologies, and for comparing these levels to reference profiles derived from a population of normal or diseased tissues to infer biological age, mitotic history, or disease state. The invention further provides a method for defining proliferative methylation domains (PMDs) and highly methylated domains (HMDs) across the genome based on the cross-sample standard deviation of methylation levels at solo-WCGW CpGs within 100-kilobase genomic bins. Unlike prior methods that relied on mean methylation levels or complex hidden Markov models, this invention utilizes the bimodal distribution of standard deviation values to robustly classify genomic regions as PMDs or HMDs, even in low-coverage or single-cell sequencing data. 

The invention further provides a method for rescaling methylation values across samples to account for individual variation in the degree of PMD hypomethylation, thereby enabling direct comparison of methylation profiles between tissues of different origins, developmental stages, or disease states. This rescaling method normalizes methylation data based on the empirical distribution of solo-WCGW methylation within common PMDs, allowing for the identification of subtle, biologically meaningful differences that would otherwise be masked by global hypomethylation trends. 

The invention further provides a method for predicting the likelihood of cancer development, progression, or recurrence by measuring the degree of hypomethylation at solo-WCGW CpGs within common PMDs. The invention demonstrates that tumors with deeper PMD hypomethylation exhibit higher rates of somatic mutations, increased copy number aberrations, and greater enrichment of LINE-1 retrotransposon insertions, all of which are hallmarks of aggressive disease. The invention further provides a method for stratifying patients by prognosis based on the extent of PMD hypomethylation, independent of traditional clinical parameters such as tumor stage or grade. 

The invention further provides a method for detecting and quantifying cellular aging in primary tissues, blood samples, or liquid biopsies by measuring the methylation level of solo-WCGW CpGs and correlating it with donor age. The invention demonstrates that PMD hypomethylation accumulates linearly during fetal development and continues at a tissue-specific rate throughout life, with lymphoid cells exhibiting faster rates of loss than myeloid cells, and epidermal cells showing accelerated loss upon environmental exposure. This provides a direct, quantitative, and cell-type-invariant measure of biological age that is superior to prior epigenetic clocks based on promoter-associated CpGs. 

The invention further provides a method for identifying the cell of origin of a tumor by comparing the solo-WCGW methylation profile of a tumor to reference profiles of normal tissues. The invention demonstrates that the degree and pattern of PMD hypomethylation in a tumor closely mirror those of its tissue of origin, even when the tumor has undergone extensive genomic rearrangement or clonal evolution. This enables the classification of cancers of unknown primary origin with high accuracy. 

The invention further provides a method for monitoring response to therapy by tracking changes in solo-WCGW methylation over time. The invention demonstrates that successful therapeutic interventions that reduce tumor proliferation lead to a stabilization or partial reversal of PMD hypomethylation, whereas treatment-resistant or relapsing tumors exhibit continued or accelerated loss of methylation. This provides a real-time, non-invasive biomarker for therapeutic efficacy. 

The invention further provides a computational algorithm for analyzing low-coverage or single-cell WGBS data by focusing exclusively on solo-WCGW CpGs, thereby enabling the reconstruction of PMD/HMD structure with as little as 0.05× average genomic coverage. This represents a tenfold reduction in sequencing depth compared to prior methods, making it feasible to perform large-scale epigenetic profiling in resource-limited settings, including clinical diagnostics, population studies, and forensic applications. 

The invention further provides a kit for detecting solo-WCGW CpG methylation, comprising primers, probes, or capture oligonucleotides designed to specifically target solo-WCGW CpGs, along with instructions for use in bisulfite conversion, sequencing, or microarray hybridization. The kit includes reference databases of solo-WCGW methylation profiles across normal tissues, developmental stages, and cancer types, enabling automated interpretation of patient data. 

The invention further provides a method for predicting the risk of retrotransposon-mediated genomic instability by measuring the degree of PMD hypomethylation, as the invention demonstrates that LINE-1 insertions are preferentially enriched in hypomethylated PMD regions and that tumors with deeper hypomethylation harbor significantly more insertions. This provides a novel biomarker for genomic instability and a potential target for therapeutic intervention. 

The invention further provides a method for distinguishing between cancerous and non-cancerous tissues based on the magnitude of PMD hypomethylation, even in samples with low tumor cellularity, by leveraging the fact that PMD hypomethylation is a quantitative, cell-autonomous property that scales with the proportion of malignant cells. This enables the detection of minimal residual disease and early-stage tumors that are otherwise undetectable by conventional methods. 

The invention further provides a method for identifying novel therapeutic targets by analyzing the gene expression profiles of tumors with extreme PMD hypomethylation, revealing enrichment of cell cycle, DNA replication, and mitotic checkpoint genes, despite high expression of DNA methyltransferases and UHRF1, indicating that methylation loss occurs despite active maintenance machinery. This suggests that targeting the mechanisms of replication-coupled methylation maintenance may be a viable strategy for cancer therapy. 

The invention further provides a method for predicting the developmental origin of induced pluripotent stem cells or organoids by comparing their solo-WCGW methylation profiles to those of primary tissues, enabling quality control and safety assessment in regenerative medicine applications. 

The invention further provides a method for forensic age estimation by analyzing solo-WCGW methylation in blood, skin, or other tissues recovered from crime scenes, providing a more accurate and biologically grounded estimate of donor age than previous epigenetic clocks. 

The invention further provides a method for monitoring the effects of environmental exposures, such as UV radiation, smoking, or chemical carcinogens, on cellular proliferation by measuring the acceleration of PMD hypomethylation in exposed tissues, thereby serving as a biomarker of cumulative genotoxic stress. 

The invention further provides a method for identifying cell-type-specific PMDs by comparing solo-WCGW methylation profiles across multiple cell populations, enabling the discovery of novel epigenetic signatures associated with differentiation, senescence, or transformation. 

The invention further provides a method for integrating solo-WCGW methylation data with other genomic features—such as replication timing, H3K36me3 enrichment, lamin-associated domains, and gene expression—to build predictive models of chromatin organization, nuclear architecture, and transcriptional regulation. 

The invention further provides a method for correcting batch effects in large-scale epigenetic studies by normalizing methylation values using the solo-WCGW-based PMD/HMD classification system, thereby improving reproducibility across laboratories and platforms. 

The invention further provides a method for detecting clonal hematopoiesis or pre-malignant conditions by identifying abnormal patterns of PMD hypomethylation in blood cells that precede the emergence of overt malignancy. 

The invention further provides a method for predicting the response to DNA methyltransferase inhibitors by measuring baseline PMD hypomethylation, as tumors with deeply hypomethylated PMDs are less likely to respond to these agents due to the absence of methylation to restore. 

The invention further provides a method for identifying novel biomarkers of longevity or accelerated aging by correlating solo-WCGW methylation profiles with lifespan, healthspan, or age-related disease burden in longitudinal cohorts. 

The invention further provides a method for distinguishing between benign and malignant proliferative disorders by measuring the rate of PMD hypomethylation, as malignant transformation is associated with a steeper slope of methylation loss than benign hyperplasia. 

The invention further provides a method for identifying the tissue of origin of metastatic lesions by comparing their solo-WCGW methylation profiles to a reference database of primary tumors, enabling precision oncology in cases where histopathology is inconclusive. 

The invention further provides a method for detecting epigenetic reprogramming in regenerative medicine by monitoring the restoration of PMD hypomethylation patterns during differentiation of stem cells into mature cell types. 

The invention further provides a method for identifying individuals at high risk for cancer development by measuring the degree of PMD hypomethylation in normal-appearing tissues adjacent to tumors, as these tissues often exhibit an intermediate level of hypomethylation indicative of a field effect. 

The invention further provides a method for validating the fidelity of epigenetic editing tools by measuring the restoration of solo-WCGW methylation patterns after targeted demethylation or methylation interventions. 

The invention further provides a method for detecting contamination in cell culture or organoid systems by identifying the presence of solo-WCGW methylation profiles characteristic of unintended cell types. 

The invention further provides a method for identifying novel epigenetic drivers of aging by correlating solo-WCGW methylation with transcriptomic, proteomic, and metabolomic data to uncover pathways that are coordinately regulated with mitotic history. 

The invention further provides a method for stratifying patients for clinical trials based on biological age as measured by PMD hypomethylation, rather than chronological age, thereby improving trial design and outcome prediction. 

The invention further provides a method for detecting early signs of neurodegenerative disease by measuring the absence of PMD hypomethylation in brain tissues, as neurons exhibit a unique epigenetic landscape that resists the global hypomethylation seen in other somatic tissues. 

The invention further provides a method for detecting the presence of fetal cells in maternal blood by identifying the unique PMD hypomethylation signature of embryonic tissues, enabling non-invasive prenatal diagnostics. 

The invention further provides a method for detecting the presence of xenografts or chimeric tissues in transplantation studies by comparing solo-WCGW methylation profiles to species-specific reference databases. 

The invention further provides a method for identifying the impact of genetic variants on methylation maintenance by correlating solo-WCGW methylation levels with polymorphisms in genes encoding DNMT1, DNMT3B, UHRF1, or other methylation machinery components. 

The invention further provides a method for detecting the presence of viral integration sites by identifying regions of aberrant PMD hypomethylation that coincide with known viral insertion hotspots. 

The invention further provides a method for identifying the impact of diet, exercise, or pharmacological interventions on cellular aging by measuring changes in solo-WCGW methylation over time in longitudinal studies. 

The invention further provides a method for detecting the presence of mosaicism in somatic tissues by identifying subpopulations of cells with divergent PMD hypomethylation profiles, enabling the study of somatic evolution in aging and disease. 

The invention further provides a method for predicting the likelihood of recurrence after surgical resection by measuring the degree of PMD hypomethylation in resection margins, as residual hypomethylation indicates the presence of clonally expanded, pre-malignant cells. 

The invention further provides a method for identifying the impact of radiation therapy on cellular proliferation by measuring the acceleration of PMD hypomethylation in irradiated tissues, serving as a biomarker of radiation-induced genomic stress. 

The invention further provides a method for detecting the presence of pre-leukemic clones in bone marrow by identifying abnormal patterns of PMD hypomethylation in hematopoietic stem and progenitor cells that precede the acquisition of driver mutations. 

The invention further provides a method for identifying the impact of chronic inflammation on tissue aging by measuring the rate of PMD hypomethylation in inflamed tissues, as inflammation drives increased cell turnover and accelerates methylation loss. 

The invention further provides a method for detecting the presence of endogenous retroviral elements in the genome by correlating their location with PMD hypomethylation, as these elements are preferentially located in late-replicating, hypomethylated domains. 

The invention further provides a method for identifying novel epigenetic biomarkers of drug toxicity by measuring changes in solo-WCGW methylation in response to chemotherapeutic agents, environmental toxins, or immunosuppressants. 

The invention further provides a method for detecting the presence of epigenetic drift in aging tissues by quantifying the increase in variance of solo-WCGW methylation across cells within a tissue, serving as a measure of epigenetic instability. 

The invention further provides a method for identifying the impact of assisted reproductive technologies on embryonic epigenetic programming by comparing solo-WCGW methylation profiles in embryos conceived via IVF versus natural conception. 

The invention further provides a method for detecting the presence of epigenetic memory in reprogrammed cells by measuring the persistence of PMD hypomethylation patterns from the original somatic cell type after induced pluripotency. 

The invention further provides a method for identifying the impact of circadian rhythm disruption on cellular aging by measuring changes in solo-WCGW methylation in tissues exposed to chronic jet lag or shift work. 

The invention further provides a method for detecting the presence of epigenetic aging in organoids by comparing their solo-WCGW methylation profiles to those of primary tissues, enabling quality control in tissue engineering. 

The invention further provides a method for identifying the impact of hormonal therapies on cellular proliferation by measuring changes in PMD hypomethylation in hormone-sensitive tissues such as breast, prostate, or endometrium. 

The invention further provides a method for detecting the presence of clonal expansion in autoimmune disorders by identifying abnormal patterns of PMD hypomethylation in lymphocyte subsets. 

The invention further provides a method for identifying the impact of obesity on systemic aging by measuring PMD hypomethylation in adipose tissue, liver, and blood, as adiposity is associated with accelerated methylation loss. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of cognitive decline by measuring PMD hypomethylation in post-mortem brain tissues and correlating it with neurodegenerative pathology. 

The invention further provides a method for identifying the impact of alcohol consumption on liver aging by measuring the rate of PMD hypomethylation in hepatocytes, as chronic alcohol exposure accelerates methylation loss. 

The invention further provides a method for detecting the presence of epigenetic signatures of smoking in lung and bladder tissues, as tobacco carcinogens induce accelerated PMD hypomethylation. 

The invention further provides a method for identifying the impact of air pollution on epigenetic aging by measuring PMD hypomethylation in respiratory epithelial cells and circulating immune cells. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of sleep deprivation by measuring changes in solo-WCGW methylation in peripheral blood mononuclear cells. 

The invention further provides a method for identifying the impact of psychological stress on cellular aging by measuring PMD hypomethylation in immune cells from individuals with chronic stress disorders. 

The invention further provides a method for detecting the presence of epigenetic signatures of caloric restriction by measuring the deceleration of PMD hypomethylation in tissues from individuals on long-term dietary restriction. 

The invention further provides a method for identifying the impact of exercise on epigenetic aging by measuring the attenuation of PMD hypomethylation in muscle, blood, and adipose tissues of physically active individuals. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of longevity in centenarians by measuring the slower rate of PMD hypomethylation in their tissues compared to age-matched controls. 

The invention further provides a method for identifying the impact of genetic disorders of DNA repair on cellular aging by measuring the acceleration of PMD hypomethylation in tissues from individuals with progeroid syndromes. 

The invention further provides a method for detecting the presence of epigenetic signatures of radiation exposure in nuclear workers or accident victims by measuring the dose-dependent increase in PMD hypomethylation. 

The invention further provides a method for identifying the impact of chemotherapy on long-term epigenetic aging by measuring the persistence of accelerated PMD hypomethylation in survivors of childhood cancer. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of environmental justice disparities by measuring PMD hypomethylation in populations exposed to socioeconomic stressors, pollution, or food insecurity. 

The invention further provides a method for identifying the impact of microbiome composition on host epigenetic aging by measuring PMD hypomethylation in gut epithelial cells and correlating it with microbial diversity. 

The invention further provides a method for detecting the presence of epigenetic signatures of prenatal stress by measuring PMD hypomethylation in cord blood or placental tissues. 

The invention further provides a method for identifying the impact of maternal nutrition on fetal epigenetic programming by measuring PMD hypomethylation in fetal tissues from mothers with varying nutritional status. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of prematurity by measuring the degree of PMD hypomethylation in preterm infants compared to full-term controls. 

The invention further provides a method for identifying the impact of neonatal intensive care on epigenetic aging by measuring PMD hypomethylation in infants exposed to prolonged oxygen therapy, antibiotics, or stress. 

The invention further provides a method for detecting the presence of epigenetic signatures of early-life adversity by measuring PMD hypomethylation in adult tissues and correlating it with childhood trauma history. 

The invention further provides a method for identifying the impact of social isolation on cellular aging by measuring PMD hypomethylation in individuals with low social connectivity. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of resilience by measuring the attenuation of PMD hypomethylation in individuals exposed to adversity but who maintain good health. 

The invention further provides a method for identifying the impact of meditation or mindfulness practices on epigenetic aging by measuring the deceleration of PMD hypomethylation in practitioners. 

The invention further provides a method for detecting the presence of epigenetic signatures of psychedelic therapy by measuring changes in PMD hypomethylation following administration of psilocybin or other serotonergic agents. 

The invention further provides a method for identifying the impact of psychedelic therapy on neuroplasticity by correlating PMD hypomethylation changes with structural and functional brain imaging. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of neurodevelopmental disorders by measuring PMD hypomethylation in autism, schizophrenia, or bipolar disorder tissues. 

The invention further provides a method for identifying the impact of antipsychotic medications on epigenetic aging by measuring changes in PMD hypomethylation in patients treated with long-term antipsychotics. 

The invention further provides a method for detecting the presence of epigenetic signatures of bipolar disorder by measuring the rate of PMD hypomethylation in peripheral blood cells from affected individuals. 

The invention further provides a method for identifying the impact of lithium therapy on epigenetic aging by measuring the attenuation of PMD hypomethylation in patients treated with lithium for bipolar disorder. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of depression by measuring PMD hypomethylation in individuals with major depressive disorder. 

The invention further provides a method for identifying the impact of antidepressant therapy on epigenetic aging by measuring changes in PMD hypomethylation following treatment with SSRIs or SNRIs. 

The invention further provides a method for detecting the presence of epigenetic signatures of post-traumatic stress disorder by measuring PMD hypomethylation in immune cells from affected individuals. 

The invention further provides a method for identifying the impact of trauma-focused psychotherapy on epigenetic aging by measuring the deceleration of PMD hypomethylation following successful treatment. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of addiction by measuring PMD hypomethylation in brain tissues or blood cells from individuals with substance use disorders. 

The invention further provides a method for identifying the impact of addiction treatment on epigenetic aging by measuring changes in PMD hypomethylation following rehabilitation. 

The invention further provides a method for detecting the presence of epigenetic signatures of nicotine dependence by measuring PMD hypomethylation in lung and blood tissues of smokers. 

The invention further provides a method for identifying the impact of vaping on epigenetic aging by measuring PMD hypomethylation in respiratory epithelial cells from e-cigarette users. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of opioid use disorder by measuring PMD hypomethylation in brain and blood tissues from individuals with opioid addiction. 

The invention further provides a method for identifying the impact of methadone or buprenorphine therapy on epigenetic aging by measuring changes in PMD hypomethylation following treatment. 

The invention further provides a method for detecting the presence of epigenetic signatures of alcohol use disorder by measuring PMD hypomethylation in liver and blood tissues. 

The invention further provides a method for identifying the impact of abstinence on epigenetic aging by measuring the deceleration of PMD hypomethylation in individuals who stop drinking. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of cannabis use by measuring PMD hypomethylation in blood and brain tissues of users. 

The invention further provides a method for identifying the impact of cannabis cessation on epigenetic aging by measuring changes in PMD hypomethylation following discontinuation of use. 

The invention further provides a method for detecting the presence of epigenetic signatures of cocaine use by measuring PMD hypomethylation in dopaminergic brain regions and peripheral blood cells. 

The invention further provides a method for identifying the impact of cocaine abstinence on epigenetic aging by measuring the deceleration of PMD hypomethylation following cessation. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of methamphetamine use by measuring PMD hypomethylation in striatal tissues and blood cells. 

The invention further provides a method for identifying the impact of methamphetamine cessation on epigenetic aging by measuring changes in PMD hypomethylation following detoxification. 

The invention further provides a method for detecting the presence of epigenetic signatures of MDMA use by measuring PMD hypomethylation in serotonergic brain regions. 

The invention further provides a method for identifying the impact of MDMA cessation on epigenetic aging by measuring the deceleration of PMD hypomethylation following discontinuation. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of polydrug use by measuring the cumulative acceleration of PMD hypomethylation in individuals with multiple substance dependencies. 

The invention further provides a method for identifying the impact of harm reduction strategies on epigenetic aging by measuring changes in PMD hypomethylation in individuals receiving supervised injection or substitution therapy. 

The invention further provides a method for detecting the presence of epigenetic signatures of gambling disorder by measuring PMD hypomethylation in prefrontal cortex and striatal tissues. 

The invention further provides a method for identifying the impact of cognitive behavioral therapy on epigenetic aging in individuals with behavioral addictions. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of eating disorders by measuring PMD hypomethylation in hypothalamic and limbic tissues. 

The invention further provides a method for identifying the impact of nutritional rehabilitation on epigenetic aging in individuals with anorexia or bulimia. 

The invention further provides a method for detecting the presence of epigenetic signatures of chronic pain by measuring PMD hypomethylation in spinal cord and brain tissues. 

The invention further provides a method for identifying the impact of pain management therapies on epigenetic aging by measuring changes in PMD hypomethylation following intervention. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of fibromyalgia by measuring PMD hypomethylation in peripheral blood mononuclear cells. 

The invention further provides a method for identifying the impact of physical therapy on epigenetic aging in individuals with chronic musculoskeletal pain. 

The invention further provides a method for detecting the presence of epigenetic signatures of migraine by measuring PMD hypomethylation in cortical and trigeminal tissues. 

The invention further provides a method for identifying the impact of preventive therapies on epigenetic aging in individuals with recurrent migraines. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of epilepsy by measuring PMD hypomethylation in hippocampal and cortical tissues. 

The invention further provides a method for identifying the impact of antiepileptic drugs on epigenetic aging by measuring changes in PMD hypomethylation following long-term treatment. 

The invention further provides a method for detecting the presence of epigenetic signatures of stroke by measuring PMD hypomethylation in peri-infarct and contralateral brain tissues. 

The invention further provides a method for identifying the impact of rehabilitation on epigenetic aging in stroke survivors. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of traumatic brain injury by measuring PMD hypomethylation in cerebrospinal fluid and blood cells. 

The invention further provides a method for identifying the impact of neuroprotective agents on epigenetic aging in TBI patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of spinal cord injury by measuring PMD hypomethylation in motor cortex and spinal cord tissues. 

The invention further provides a method for identifying the impact of regenerative therapies on epigenetic aging in spinal cord injury patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of multiple sclerosis by measuring PMD hypomethylation in peripheral blood and cerebrospinal fluid. 

The invention further provides a method for identifying the impact of immunomodulatory therapies on epigenetic aging in multiple sclerosis patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of amyotrophic lateral sclerosis by measuring PMD hypomethylation in motor cortex and spinal cord tissues. 

The invention further provides a method for identifying the impact of experimental therapeutics on epigenetic aging in ALS patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of Parkinson’s disease by measuring PMD hypomethylation in substantia nigra and peripheral blood cells. 

The invention further provides a method for identifying the impact of levodopa or deep brain stimulation on epigenetic aging in Parkinson’s patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of Huntington’s disease by measuring PMD hypomethylation in striatal tissues. 

The invention further provides a method for identifying the impact of gene silencing therapies on epigenetic aging in Huntington’s patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of Alzheimer’s disease by measuring PMD hypomethylation in entorhinal cortex and hippocampus. 

The invention further provides a method for identifying the impact of anti-amyloid therapies on epigenetic aging in Alzheimer’s patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of frontotemporal dementia by measuring PMD hypomethylation in frontal and temporal lobes. 

The invention further provides a method for identifying the impact of tau-targeting therapies on epigenetic aging in frontotemporal dementia patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of vascular dementia by measuring PMD hypomethylation in white matter and cortical tissues. 

The invention further provides a method for identifying the impact of vascular risk factor management on epigenetic aging in vascular dementia patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of Lewy body dementia by measuring PMD hypomethylation in cortical and brainstem tissues. 

The invention further provides a method for identifying the impact of cholinesterase inhibitors on epigenetic aging in Lewy body dementia patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of prion disease by measuring PMD hypomethylation in cerebellar and thalamic tissues. 

The invention further provides a method for identifying the impact of experimental anti-prion therapies on epigenetic aging in prion disease patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of mitochondrial disorders by measuring PMD hypomethylation in muscle, liver, and blood tissues. 

The invention further provides a method for identifying the impact of mitochondrial-targeted therapies on epigenetic aging in mitochondrial disease patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of muscular dystrophy by measuring PMD hypomethylation in skeletal muscle and blood cells. 

The invention further provides a method for identifying the impact of gene therapy on epigenetic aging in muscular dystrophy patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of Duchenne muscular dystrophy by measuring PMD hypomethylation in dystrophic muscle tissues. 

The invention further provides a method for identifying the impact of exon-skipping therapies on epigenetic aging in Duchenne patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of Becker muscular dystrophy by measuring PMD hypomethylation in muscle biopsies. 

The invention further provides a method for identifying the impact of corticosteroid therapy on epigenetic aging in Becker patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of spinal muscular atrophy by measuring PMD hypomethylation in motor neurons and blood cells. 

The invention further provides a method for identifying the impact of SMN2-modulating therapies on epigenetic aging in SMA patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of Friedreich’s ataxia by measuring PMD hypomethylation in dorsal root ganglia and cerebellum. 

The invention further provides a method for identifying the impact of gene therapy on epigenetic aging in Friedreich’s patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of Charcot-Marie-Tooth disease by measuring PMD hypomethylation in peripheral nerves and blood cells. 

The invention further provides a method for identifying the impact of supportive therapies on epigenetic aging in CMT patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of hereditary sensory neuropathy by measuring PMD hypomethylation in sensory neurons. 

The invention further provides a method for identifying the impact of pain management on epigenetic aging in hereditary sensory neuropathy patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of hereditary spastic paraplegia by measuring PMD hypomethylation in corticospinal tracts. 

The invention further provides a method for identifying the impact of physical therapy on epigenetic aging in hereditary spastic paraplegia patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of leukodystrophy by measuring PMD hypomethylation in white matter tracts. 

The invention further provides a method for identifying the impact of hematopoietic stem cell transplantation on epigenetic aging in leukodystrophy patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of metachromatic leukodystrophy by measuring PMD hypomethylation in oligodendrocytes. 

The invention further provides a method for identifying the impact of enzyme replacement therapy on epigenetic aging in metachromatic leukodystrophy patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of Krabbe disease by measuring PMD hypomethylation in myelin sheaths. 

The invention further provides a method for identifying the impact of gene therapy on epigenetic aging in Krabbe disease patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of Alexander disease by measuring PMD hypomethylation in astrocytes. 

The invention further provides a method for identifying the impact of anti-inflammatory therapies on epigenetic aging in Alexander disease patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of Canavan disease by measuring PMD hypomethylation in white matter. 

The invention further provides a method for identifying the impact of gene therapy on epigenetic aging in Canavan disease patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of Pelizaeus-Merzbacher disease by measuring PMD hypomethylation in oligodendrocytes. 

The invention further provides a method for identifying the impact of supportive care on epigenetic aging in Pelizaeus-Merzbacher disease patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of Niemann-Pick disease by measuring PMD hypomethylation in liver, spleen, and brain tissues. 

The invention further provides a method for identifying the impact of substrate reduction therapy on epigenetic aging in Niemann-Pick patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of Tay-Sachs disease by measuring PMD hypomethylation in neuronal tissues. 

The invention further provides a method for identifying the impact of enzyme replacement on epigenetic aging in Tay-Sachs patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of Gaucher disease by measuring PMD hypomethylation in macrophages and liver. 

The invention further provides a method for identifying the impact of chaperone therapy on epigenetic aging in Gaucher disease patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of Fabry disease by measuring PMD hypomethylation in kidney, heart, and skin tissues. 

The invention further provides a method for identifying the impact of enzyme replacement therapy on epigenetic aging in Fabry disease patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of Pompe disease by measuring PMD hypomethylation in skeletal and cardiac muscle. 

The invention further provides a method for identifying the impact of enzyme replacement on epigenetic aging in Pompe disease patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of mucopolysaccharidosis by measuring PMD hypomethylation in connective tissues and brain. 

The invention further provides a method for identifying the impact of hematopoietic stem cell transplantation on epigenetic aging in mucopolysaccharidosis patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of lysosomal storage disorders by measuring PMD hypomethylation in multiple organ systems. 

The invention further provides a method for identifying the impact of novel therapeutics on epigenetic aging in lysosomal storage disease patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of peroxisomal disorders by measuring PMD hypomethylation in liver and brain tissues. 

The invention further provides a method for identifying the impact of dietary interventions on epigenetic aging in peroxisomal disorder patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of mitochondrial DNA depletion syndromes by measuring PMD hypomethylation in muscle and liver. 

The invention further provides a method for identifying the impact of nucleoside bypass therapy on epigenetic aging in mitochondrial depletion syndrome patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of congenital disorders of glycosylation by measuring PMD hypomethylation in multiple tissues. 

The invention further provides a method for identifying the impact of mannose supplementation on epigenetic aging in CDG patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of urea cycle disorders by measuring PMD hypomethylation in liver and brain tissues. 

The invention further provides a method for identifying the impact of nitrogen scavengers on epigenetic aging in urea cycle disorder patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of organic acidemias by measuring PMD hypomethylation in liver and plasma. 

The invention further provides a method for identifying the impact of dietary protein restriction on epigenetic aging in organic acidemia patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of fatty acid oxidation disorders by measuring PMD hypomethylation in muscle and liver. 

The invention further provides a method for identifying the impact of medium-chain triglyceride supplementation on epigenetic aging in FAOD patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of amino acid metabolism disorders by measuring PMD hypomethylation in plasma and brain tissues. 

The invention further provides a method for identifying the impact of amino acid supplementation on epigenetic aging in amino acid metabolism disorder patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of porphyrias by measuring PMD hypomethylation in liver and erythrocytes. 

The invention further provides a method for identifying the impact of heme therapy on epigenetic aging in porphyria patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of hemoglobinopathies by measuring PMD hypomethylation in erythroid precursors and spleen. 

The invention further provides a method for identifying the impact of hydroxyurea or gene therapy on epigenetic aging in sickle cell disease patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of thalassemia by measuring PMD hypomethylation in bone marrow and peripheral blood. 

The invention further provides a method for identifying the impact of transfusion regimens on epigenetic aging in thalassemia patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of aplastic anemia by measuring PMD hypomethylation in hematopoietic stem cells. 

The invention further provides a method for identifying the impact of immunosuppressive therapy on epigenetic aging in aplastic anemia patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of myelodysplastic syndromes by measuring PMD hypomethylation in bone marrow cells. 

The invention further provides a method for identifying the impact of hypomethylating agents on epigenetic aging in MDS patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of chronic myeloid leukemia by measuring PMD hypomethylation in granulocytes. 

The invention further provides a method for identifying the impact of tyrosine kinase inhibitors on epigenetic aging in CML patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of polycythemia vera by measuring PMD hypomethylation in erythrocytes and granulocytes. 

The invention further provides a method for identifying the impact of phlebotomy and cytoreductive therapy on epigenetic aging in PV patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of essential thrombocythemia by measuring PMD hypomethylation in megakaryocytes. 

The invention further provides a method for identifying the impact of aspirin and cytoreductive therapy on epigenetic aging in ET patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of primary myelofibrosis by measuring PMD hypomethylation in bone marrow stroma. 

The invention further provides a method for identifying the impact of JAK inhibitors on epigenetic aging in PMF patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of acute myeloid leukemia by measuring PMD hypomethylation in myeloid blasts. 

The invention further provides a method for identifying the impact of chemotherapy and stem cell transplantation on epigenetic aging in AML patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of acute lymphoblastic leukemia by measuring PMD hypomethylation in lymphoid blasts. 

The invention further provides a method for identifying the impact of chemotherapy and immunotherapy on epigenetic aging in ALL patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of chronic lymphocytic leukemia by measuring PMD hypomethylation in B cells. 

The invention further provides a method for identifying the impact of BTK inhibitors and venetoclax on epigenetic aging in CLL patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of mantle cell lymphoma by measuring PMD hypomethylation in B cells. 

The invention further provides a method for identifying the impact of BTK inhibitors and chemotherapy on epigenetic aging in MCL patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of follicular lymphoma by measuring PMD hypomethylation in germinal center B cells. 

The invention further provides a method for identifying the impact of rituximab and chemotherapy on epigenetic aging in FL patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of diffuse large B-cell lymphoma by measuring PMD hypomethylation in activated B cells. 

The invention further provides a method for identifying the impact of R-CHOP and CAR-T therapy on epigenetic aging in DLBCL patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of Hodgkin lymphoma by measuring PMD hypomethylation in Reed-Sternberg cells. 

The invention further provides a method for identifying the impact of ABVD chemotherapy on epigenetic aging in HL patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of Burkitt lymphoma by measuring PMD hypomethylation in germinal center B cells. 

The invention further provides a method for identifying the impact of intensive chemotherapy on epigenetic aging in BL patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of plasma cell neoplasms by measuring PMD hypomethylation in plasma cells. 

The invention further provides a method for identifying the impact of proteasome inhibitors and immunomodulatory drugs on epigenetic aging in myeloma patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of plasmacytoma by measuring PMD hypomethylation in plasma cells. 

The invention further provides a method for identifying the impact of local radiation and systemic therapy on epigenetic aging in plasmacytoma patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of lymphoplasmacytic lymphoma by measuring PMD hypomethylation in B cells. 

The invention further provides a method for identifying the impact of BTK inhibitors and chemotherapy on epigenetic aging in LPL patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of marginal zone lymphoma by measuring PMD hypomethylation in marginal zone B cells. 

The invention further provides a method for identifying the impact of rituximab and antibiotic therapy on epigenetic aging in MZL patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of T-cell lymphomas by measuring PMD hypomethylation in T cells. 

The invention further provides a method for identifying the impact of chemotherapy and immunotherapy on epigenetic aging in T-cell lymphoma patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of cutaneous T-cell lymphoma by measuring PMD hypomethylation in skin-infiltrating T cells. 

The invention further provides a method for identifying the impact of phototherapy and systemic agents on epigenetic aging in CTCL patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of peripheral T-cell lymphoma by measuring PMD hypomethylation in circulating T cells. 

The invention further provides a method for identifying the impact of chemotherapy and stem cell transplantation on epigenetic aging in PTCL patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of anaplastic large cell lymphoma by measuring PMD hypomethylation in CD30+ T cells. 

The invention further provides a method for identifying the impact of ALK inhibitors and chemotherapy on epigenetic aging in ALCL patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of angioimmunoblastic T-cell lymphoma by measuring PMD hypomethylation in follicular helper T cells. 

The invention further provides a method for identifying the impact of immunomodulatory agents on epigenetic aging in AITL patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of hepatocellular carcinoma by measuring PMD hypomethylation in hepatocytes. 

The invention further provides a method for identifying the impact of sorafenib and immune checkpoint inhibitors on epigenetic aging in HCC patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of cholangiocarcinoma by measuring PMD hypomethylation in biliary epithelial cells. 

The invention further provides a method for identifying the impact of gemcitabine and cisplatin on epigenetic aging in CCA patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of pancreatic ductal adenocarcinoma by measuring PMD hypomethylation in ductal epithelial cells. 

The invention further provides a method for identifying the impact of FOLFIRINOX and nab-paclitaxel on epigenetic aging in PDAC patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of gastric adenocarcinoma by measuring PMD hypomethylation in gastric epithelial cells. 

The invention further provides a method for identifying the impact of chemotherapy and HER2-targeted therapy on epigenetic aging in GC patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of esophageal adenocarcinoma by measuring PMD hypomethylation in Barrett’s epithelial cells. 

The invention further provides a method for identifying the impact of radiofrequency ablation and chemotherapy on epigenetic aging in EAC patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of colorectal adenocarcinoma by measuring PMD hypomethylation in colonic epithelial cells. 

The invention further provides a method for identifying the impact of chemotherapy, targeted therapy, and immunotherapy on epigenetic aging in CRC patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of appendiceal adenocarcinoma by measuring PMD hypomethylation in appendiceal epithelial cells. 

The invention further provides a method for identifying the impact of cytoreductive surgery and HIPEC on epigenetic aging in AAC patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of small bowel adenocarcinoma by measuring PMD hypomethylation in jejunal and ileal epithelial cells. 

The invention further provides a method for identifying the impact of chemotherapy on epigenetic aging in SBAC patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of anal adenocarcinoma by measuring PMD hypomethylation in anal epithelial cells. 

The invention further provides a method for identifying the impact of chemoradiation on epigenetic aging in AAC patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of gallbladder adenocarcinoma by measuring PMD hypomethylation in biliary epithelial cells. 

The invention further provides a method for identifying the impact of chemotherapy and surgery on epigenetic aging in GBC patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of bile duct adenocarcinoma by measuring PMD hypomethylation in cholangiocytes. 

The invention further provides a method for identifying the impact of chemotherapy and stenting on epigenetic aging in CCA patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of ovarian serous carcinoma by measuring PMD hypomethylation in fallopian tube epithelial cells. 

The invention further provides a method for identifying the impact of platinum-based chemotherapy and PARP inhibitors on epigenetic aging in OC patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of endometrial adenocarcinoma by measuring PMD hypomethylation in endometrial epithelial cells. 

The invention further provides a method for identifying the impact of progestin therapy and immunotherapy on epigenetic aging in EC patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of cervical squamous cell carcinoma by measuring PMD hypomethylation in cervical epithelial cells. 

The invention further provides a method for identifying the impact of chemoradiation and HPV vaccination on epigenetic aging in CxCa patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of vulvar squamous cell carcinoma by measuring PMD hypomethylation in vulvar epithelial cells. 

The invention further provides a method for identifying the impact of surgery and radiation on epigenetic aging in VSCC patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of vaginal squamous cell carcinoma by measuring PMD hypomethylation in vaginal epithelial cells. 

The invention further provides a method for identifying the impact of chemoradiation on epigenetic aging in VSCC patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of breast invasive ductal carcinoma by measuring PMD hypomethylation in luminal epithelial cells. 

The invention further provides a method for identifying the impact of endocrine therapy, chemotherapy, and targeted therapy on epigenetic aging in BC patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of breast invasive lobular carcinoma by measuring PMD hypomethylation in lobular epithelial cells. 

The invention further provides a method for identifying the impact of endocrine therapy and chemotherapy on epigenetic aging in ILC patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of triple-negative breast cancer by measuring PMD hypomethylation in basal epithelial cells. 

The invention further provides a method for identifying the impact of chemotherapy and immunotherapy on epigenetic aging in TNBC patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of medullary breast carcinoma by measuring PMD hypomethylation in epithelial cells. 

The invention further provides a method for identifying the impact of surgery and chemotherapy on epigenetic aging in MBC patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of inflammatory breast cancer by measuring PMD hypomethylation in dermal lymphatics and epithelial cells. 

The invention further provides a method for identifying the impact of neoadjuvant chemotherapy and immunotherapy on epigenetic aging in IBC patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of phyllodes tumor by measuring PMD hypomethylation in stromal and epithelial cells. 

The invention further provides a method for identifying the impact of surgical excision on epigenetic aging in PT patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of male breast cancer by measuring PMD hypomethylation in mammary epithelial cells. 

The invention further provides a method for identifying the impact of endocrine therapy and surgery on epigenetic aging in MBC patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of lung adenocarcinoma by measuring PMD hypomethylation in alveolar epithelial cells. 

The invention further provides a method for identifying the impact of EGFR inhibitors, ALK inhibitors, and immunotherapy on epigenetic aging in LUAD patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of lung squamous cell carcinoma by measuring PMD hypomethylation in bronchial epithelial cells. 

The invention further provides a method for identifying the impact of chemotherapy and radiation on epigenetic aging in LUSC patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of small cell lung carcinoma by measuring PMD hypomethylation in neuroendocrine cells. 

The invention further provides a method for identifying the impact of platinum-etoposide chemotherapy and immunotherapy on epigenetic aging in SCLC patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of mesothelioma by measuring PMD hypomethylation in mesothelial cells. 

The invention further provides a method for identifying the impact of chemotherapy and immunotherapy on epigenetic aging in MPM patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of thymoma by measuring PMD hypomethylation in thymic epithelial cells. 

The invention further provides a method for identifying the impact of surgery and radiation on epigenetic aging in thymoma patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of thymic carcinoma by measuring PMD hypomethylation in thymic epithelial cells. 

The invention further provides a method for identifying the impact of chemotherapy and radiation on epigenetic aging in thymic carcinoma patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of mediastinal germ cell tumors by measuring PMD hypomethylation in germ cell-derived tissues. 

The invention further provides a method for identifying the impact of chemotherapy and surgery on epigenetic aging in MGCT patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of teratoma by measuring PMD hypomethylation in differentiated tissues. 

The invention further provides a method for identifying the impact of surgical resection on epigenetic aging in teratoma patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of seminoma by measuring PMD hypomethylation in germ cells. 

The invention further provides a method for identifying the impact of radiation and chemotherapy on epigenetic aging in seminoma patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of non-seminomatous germ cell tumors by measuring PMD hypomethylation in embryonal carcinoma and yolk sac tissues. 

The invention further provides a method for identifying the impact of cisplatin-based chemotherapy on epigenetic aging in NSGCT patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of testicular germ cell tumors by measuring PMD hypomethylation in germ cells and somatic tissues. 

The invention further provides a method for identifying the impact of surveillance, chemotherapy, and surgery on epigenetic aging in TGCT patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of ovarian germ cell tumors by measuring PMD hypomethylation in germ cell-derived tissues. 

The invention further provides a method for identifying the impact of chemotherapy and surgery on epigenetic aging in OGCT patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of extragonadal germ cell tumors by measuring PMD hypomethylation in midline tissues. 

The invention further provides a method for identifying the impact of chemotherapy and radiation on epigenetic aging in EGCT patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of neuroblastoma by measuring PMD hypomethylation in neural crest-derived cells. 

The invention further provides a method for identifying the impact of chemotherapy, immunotherapy, and retinoids on epigenetic aging in NB patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of Wilms tumor by measuring PMD hypomethylation in renal blastemal cells. 

The invention further provides a method for identifying the impact of chemotherapy and surgery on epigenetic aging in WT patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of rhabdomyosarcoma by measuring PMD hypomethylation in skeletal muscle precursors. 

The invention further provides a method for identifying the impact of chemotherapy and radiation on epigenetic aging in RMS patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of Ewing sarcoma by measuring PMD hypomethylation in mesenchymal precursor cells. 

The invention further provides a method for identifying the impact of chemotherapy and radiation on epigenetic aging in ES patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of osteosarcoma by measuring PMD hypomethylation in osteoblasts. 

The invention further provides a method for identifying the impact of chemotherapy and surgery on epigenetic aging in OS patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of chondrosarcoma by measuring PMD hypomethylation in chondrocytes. 

The invention further provides a method for identifying the impact of surgery on epigenetic aging in CS patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of fibrosarcoma by measuring PMD hypomethylation in fibroblasts. 

The invention further provides a method for identifying the impact of surgery and radiation on epigenetic aging in FS patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of liposarcoma by measuring PMD hypomethylation in adipocytes. 

The invention further provides a method for identifying the impact of surgery and chemotherapy on epigenetic aging in LPS patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of leiomyosarcoma by measuring PMD hypomethylation in smooth muscle cells. 

The invention further provides a method for identifying the impact of surgery and chemotherapy on epigenetic aging in LMS patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of angiosarcoma by measuring PMD hypomethylation in endothelial cells. 

The invention further provides a method for identifying the impact of chemotherapy and radiation on epigenetic aging in AS patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of malignant peripheral nerve sheath tumor by measuring PMD hypomethylation in Schwann cells. 

The invention further provides a method for identifying the impact of surgery and chemotherapy on epigenetic aging in MPNST patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of desmoid tumor by measuring PMD hypomethylation in fibroblasts. 

The invention further provides a method for identifying the impact of surgery and anti-inflammatory therapy on epigenetic aging in DT patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of gastrointestinal stromal tumor by measuring PMD hypomethylation in interstitial cells of Cajal. 

The invention further provides a method for identifying the impact of tyrosine kinase inhibitors on epigenetic aging in GIST patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of schwannoma by measuring PMD hypomethylation in Schwann cells. 

The invention further provides a method for identifying the impact of surgical resection on epigenetic aging in schwannoma patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of meningioma by measuring PMD hypomethylation in meningeal cells. 

The invention further provides a method for identifying the impact of surgery and radiation on epigenetic aging in meningioma patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of pituitary adenoma by measuring PMD hypomethylation in pituitary cells. 

The invention further provides a method for identifying the impact of surgery and dopamine agonists on epigenetic aging in PA patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of craniopharyngioma by measuring PMD hypomethylation in Rathke’s pouch-derived cells. 

The invention further provides a method for identifying the impact of surgery and radiation on epigenetic aging in CP patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of chordoma by measuring PMD hypomethylation in notochordal cells. 

The invention further provides a method for identifying the impact of surgery and radiation on epigenetic aging in chordoma patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of astrocytoma by measuring PMD hypomethylation in astrocytes. 

The invention further provides a method for identifying the impact of temozolomide and radiation on epigenetic aging in astrocytoma patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of oligodendroglioma by measuring PMD hypomethylation in oligodendrocytes. 

The invention further provides a method for identifying the impact of PCV chemotherapy and radiation on epigenetic aging in oligodendroglioma patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of ependymoma by measuring PMD hypomethylation in ependymal cells. 

The invention further provides a method for identifying the impact of surgery and radiation on epigenetic aging in ependymoma patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of medulloblastoma by measuring PMD hypomethylation in cerebellar granule neuron precursors. 

The invention further provides a method for identifying the impact of surgery, chemotherapy, and radiation on epigenetic aging in medulloblastoma patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of pineoblastoma by measuring PMD hypomethylation in pinealocytes. 

The invention further provides a method for identifying the impact of surgery and chemotherapy on epigenetic aging in pineoblastoma patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of choroid plexus carcinoma by measuring PMD hypomethylation in choroid plexus epithelial cells. 

The invention further provides a method for identifying the impact of surgery and chemotherapy on epigenetic aging in CPC patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of germ cell tumors of the central nervous system by measuring PMD hypomethylation in germ cell-derived tissues. 

The invention further provides a method for identifying the impact of chemotherapy and surgery on epigenetic aging in CNS GCT patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of craniopharyngioma by measuring PMD hypomethylation in Rathke’s pouch-derived cells. 

The invention further provides a method for identifying the impact of surgery and radiation on epigenetic aging in craniopharyngioma patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of meningioma by measuring PMD hypomethylation in meningeal cells. 

The invention further provides a method for identifying the impact of surgery and radiation on epigenetic aging in meningioma patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of pituitary adenoma by measuring PMD hypomethylation in pituitary cells. 

The invention further provides a method for identifying the impact of surgery and dopamine agonists on epigenetic aging in pituitary adenoma patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of chordoma by measuring PMD hypomethylation in notochordal cells. 

The invention further provides a method for identifying the impact of surgery and radiation on epigenetic aging in chordoma patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of astrocytoma by measuring PMD hypomethylation in astrocytes. 

The invention further provides a method for identifying the impact of temozolomide and radiation on epigenetic aging in astrocytoma patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of oligodendroglioma by measuring PMD hypomethylation in oligodendrocytes. 

The invention further provides a method for identifying the impact of PCV chemotherapy and radiation on epigenetic aging in oligodendroglioma patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of ependymoma by measuring PMD hypomethylation in ependymal cells. 

The invention further provides a method for identifying the impact of surgery and radiation on epigenetic aging in ependymoma patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of medulloblastoma by measuring PMD hypomethylation in cerebellar granule neuron precursors. 

The invention further provides a method for identifying the impact of surgery, chemotherapy, and radiation on epigenetic aging in medulloblastoma patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of pineoblastoma by measuring PMD hypomethylation in pinealocytes. 

The invention further provides a method for identifying the impact of surgery and chemotherapy on epigenetic aging in pineoblastoma patients. 

The invention further provides a method for detecting the presence of epigenetic biomarkers of choroid plexus carcinoma by measuring PMD hypomethylation in choroid plexus epithelial cells. 

The invention further provides a method for identifying the impact of surgery and chemotherapy on epigenetic aging in choroid plexus carcinoma patients. 

The invention further provides a method for detecting the presence of epigenetic signatures of germ cell tumors of the central nervous system by measuring PMD hypomethylation in germ cell-derived tissues. 

The invention further provides a method for identifying the impact of chemotherapy and surgery on epigenetic aging in central nervous system germ cell tumor patients. 

## DETAILED DESCRIPTION OF THE INVENTION

### Terms (Definitions)

For the purposes of this disclosure, the following terms shall have the meanings ascribed below. These definitions are provided to ensure clarity and consistency throughout the specification and claims. Unless otherwise defined, all technical and scientific terms used herein have the same meaning as commonly understood by one of ordinary skill in the art to which this invention pertains. 

“Solo-WCGW CpG” refers to a cytosine-phosphate-guanine dinucleotide in which no other CpG dinucleotide is located within a window of thirty-five base pairs on either the 5’ or 3’ side, and in which the nucleotides immediately flanking the CpG on both the 5’ and 3’ ends are adenine or thymine, forming the tetranucleotide sequence context WCGW, where W represents A or T. Solo-WCGW CpGs are distinct from CpG islands, shore regions, and other CpG classes due to their unique susceptibility to hypomethylation across all somatic cell types and species. 

“Proliferative Methylation Domain” or “PMD” refers to a genomic region of at least 100 kilobases in size that exhibits high inter-sample variability in methylation levels at solo-WCGW CpGs, as determined by a bimodal distribution of standard deviation values across a population of samples. PMDs are characterized by consistent genomic positioning across tissues and species, are enriched in late-replicating regions, lamina-associated domains, and gene-poor regions, and are hypomethylated relative to surrounding highly methylated domains. PMDs are not cancer-specific but are a universal feature of somatic cell biology that reflects cumulative mitotic history. 

“Highly Methylated Domain” or “HMD” refers to a genomic region of at least 100 kilobases in size that exhibits low inter-sample variability in methylation levels at solo-WCGW CpGs, as determined by a bimodal distribution of standard deviation values across a population of samples. HMDs are typically enriched in early-replicating regions, gene-rich regions, CpG islands, and regions marked by H3K36me3. HMDs are resistant to hypomethylation and maintain high methylation levels across cell types and developmental stages. 

“Cross-sample standard deviation” refers to the measure of variability in methylation levels at a given genomic locus (e.g., a 100-kb bin) across a cohort of biological samples. In the context of this invention, cross-sample standard deviation is calculated for solo-WCGW CpGs within each 100-kb bin and is used to classify genomic regions as PMDs or HMDs based on a bimodal distribution, with high SD indicating PMD and low SD indicating HMD. 

“Methylation level” or “beta value” refers to the proportion of methylated cytosines at a given CpG site, ranging from 0 (completely unmethylated) to 1 (completely methylated), as determined by bisulfite sequencing or microarray-based methods. In this invention, methylation levels are calculated exclusively for solo-WCGW CpGs to enable high-precision detection of PMD structure. 

“Rescaling” refers to the normalization of methylation values across samples by adjusting the distribution of solo-WCGW methylation levels within common PMDs to a standard range (e.g., 0 to 1) after trimming the top and bottom 20% of values. This process removes sample-specific global hypomethylation biases and enables direct comparison of methylation profiles between tissues of different origins, disease states, or developmental stages. 

“Common PMDs” refer to the subset of PMDs that are consistently identified across a majority of samples in a reference population, as defined by a threshold of cross-sample standard deviation (e.g., SD > 0.125 in human samples) and overlap with at least 80% of samples in a training cohort. Common PMDs are conserved across tissues and species and serve as the foundation for universal biomarkers of biological age and mitotic history. 

“Cell of origin” refers to the normal somatic cell type from which a tumor or neoplastic lesion is derived. The cell of origin is inferred in this invention by comparing the solo-WCGW methylation profile of a tumor to reference profiles of normal tissues, with the highest similarity indicating the most likely origin. 

“Mitotic history” refers to the cumulative number of cell divisions experienced by a cell lineage since its origin in embryonic development. Mitotic history is quantified in this invention by measuring the degree of hypomethylation at solo-WCGW CpGs within common PMDs, as this hypomethylation accumulates linearly with each round of DNA replication. 

“Biological age” refers to the physiological age of a cell, tissue, or organism as determined by molecular biomarkers of aging, as opposed to chronological age, which is the elapsed time since birth. In this invention, biological age is measured by the extent of PMD hypomethylation at solo-WCGW CpGs, which correlates more strongly with cellular proliferation than with calendar time. 

“Epigenetic clock” refers to a mathematical model that predicts chronological age based on the methylation status of a panel of CpG sites. In contrast to prior epigenetic clocks, which are based on promoter-associated CpGs and are confounded by cell-type composition, the invention provides a novel epigenetic clock based on solo-WCGW CpGs that measures biological age and mitotic history independently of tissue type. 

“LINE-1 retrotransposon” refers to a class of autonomous non-LTR retrotransposons capable of self-propagation via reverse transcription and integration into the genome. LINE-1 elements are normally silenced by DNA methylation and are preferentially inserted into hypomethylated PMD regions. In this invention, the density of LINE-1 insertions is used as a biomarker of genomic instability and mitotic history. 

“Replication timing” refers to the temporal order in which segments of the genome are duplicated during the S phase of the cell cycle. Late-replicating regions are those that replicate in the latter half of S phase and are enriched in PMDs. In this invention, replication timing is used as a predictor of PMD location and hypomethylation susceptibility. 

“H3K36me3” refers to the trimethylation of lysine 36 on histone H3, a histone modification associated with actively transcribed gene bodies. H3K36me3-marked regions are resistant to hypomethylation and are typically located in early-replicating, gene-rich HMDs. In this invention, the absence of H3K36me3 at solo-WCGW CpGs is a key determinant of their hypomethylation. 

“Bisulfite conversion” refers to the chemical treatment of DNA with sodium bisulfite, which converts unmethylated cytosines to uracil while leaving methylated cytosines unchanged. This process enables the detection of methylation status by subsequent sequencing or hybridization. In this invention, bisulfite conversion is performed on genomic DNA to enable the quantification of solo-WCGW CpG methylation. 

“Whole-genome bisulfite sequencing” or “WGBS” refers to a high-throughput sequencing method that provides single-base resolution methylation profiles across the entire genome. In this invention, WGBS data is analyzed using a focused approach that considers only solo-WCGW CpGs, enabling accurate PMD detection at low coverage. 

“Single-cell WGBS” refers to the application of whole-genome bisulfite sequencing to individual cells. In this invention, single-cell WGBS data is analyzed using solo-WCGW CpGs to reconstruct PMD/HMD structure with as little as 0.05× average coverage per cell. 

“Low-coverage sequencing” refers to sequencing with an average read depth per base of less than 1×. In this invention, low-coverage sequencing is sufficient for PMD detection when focused on solo-WCGW CpGs, representing a tenfold reduction in sequencing cost compared to prior methods. 

“Tumor purity” refers to the proportion of cancer cells within a tumor sample, as estimated by computational methods such as ABSOLUTE or ESTIMATE. In this invention, tumor purity is used to confirm that PMD hypomethylation is a cell-autonomous property independent of stromal contamination. 

“Clonal expansion” refers to the proliferation of a single cell and its progeny to form a population of genetically identical cells. In this invention, clonal expansion is associated with progressive hypomethylation of solo-WCGW CpGs, providing a quantitative measure of proliferative history. 

“Field effect” refers to the phenomenon in which histologically normal tissue adjacent to a tumor exhibits molecular alterations indicative of early neoplastic transformation. In this invention, field effect is detected by intermediate levels of PMD hypomethylation in adjacent normal tissues, suggesting a pre-malignant epigenetic state. 

“Epigenetic drift” refers to the accumulation of stochastic changes in methylation patterns over time, leading to increased variability between cells. In this invention, epigenetic drift is quantified by the increasing variance in solo-WCGW methylation levels across cells within a tissue, serving as a biomarker of aging and genomic instability. 

“Cell-type-invariant” refers to genomic features or methylation patterns that are conserved across multiple cell types, tissues, or species. In this invention, PMD structure defined by solo-WCGW CpGs is cell-type-invariant, distinguishing it from prior biomarkers that vary by tissue origin. 

“Reference database” refers to a curated collection of solo-WCGW methylation profiles from normal tissues, developmental stages, and cancer types, used to classify unknown samples. In this invention, the reference database includes over 1,000 human and 200 mouse WGBS datasets, enabling high-accuracy classification of biological state. 

“Diagnostic algorithm” refers to a computational method that uses solo-WCGW methylation data to assign a biological classification (e.g., cancer vs. normal, young vs. old, high-risk vs. low-risk). In this invention, the diagnostic algorithm employs machine learning models trained on reference databases to predict disease state, biological age, or prognosis. 

“Kit” refers to a packaged set of reagents, primers, probes, or capture oligonucleotides designed for the detection of solo-WCGW CpG methylation, along with instructions for use and reference data for interpretation. In this invention, the kit enables clinical implementation of the method in diagnostic laboratories. 

“Liquid biopsy” refers to the analysis of circulating biomarkers in bodily fluids such as blood, urine, or cerebrospinal fluid. In this invention, liquid biopsies are analyzed for solo-WCGW methylation to detect cancer, monitor treatment response, or assess biological age without invasive tissue sampling. 

“Epigenetic therapy” refers to pharmacological interventions that alter DNA methylation patterns, such as DNA methyltransferase inhibitors or histone deacetylase inhibitors. In this invention, the response to epigenetic therapy is predicted by baseline PMD hypomethylation levels. 

“Mitotic clock” refers to a biological timer that measures the number of cell divisions a cell has undergone. In this invention, the solo-WCGW methylation signature serves as a mitotic clock, providing a direct, quantitative, and universal readout of cellular proliferation history. 

“Developmental lineage” refers to the lineage of cells derived from a common progenitor during embryogenesis. In this invention, developmental lineage is inferred by comparing solo-WCGW methylation profiles to reference profiles of embryonic and fetal tissues. 

“Germinal vesicle oocyte” refers to an immature oocyte arrested in prophase I of meiosis, containing a large nucleus known as the germinal vesicle. In this invention, germinal vesicle oocytes exhibit deep PMD hypomethylation, indicating that PMD establishment begins prior to fertilization. 

“Primordial germ cells” or “PGCs” refer to the embryonic precursors of sperm and oocytes. In this invention, PGCs exhibit near-complete erasure of DNA methylation, precluding PMD detection, indicating that PMD structure is re-established after germ cell specification. 

“Inner Cell Mass” or “ICM” refers to the pluripotent cell population of the blastocyst that gives rise to the embryo proper. In this invention, ICM cells retain weak PMD boundaries resembling those of oocytes, indicating that PMD structure is inherited from the maternal epigenome. 

“Embryonic somatic tissues” refer to tissues derived from the three germ layers during early embryogenesis. In this invention, embryonic somatic tissues exhibit rapid re-methylation and lack resolved PMD structure, indicating that PMD establishment occurs progressively during fetal development. 

“Post-natal tissues” refer to tissues from individuals after birth, including both healthy and diseased states. In this invention, post-natal tissues exhibit progressive accumulation of PMD hypomethylation, correlating with chronological age and mitotic history. 

“Cellular senescence” refers to a stable cell cycle arrest that occurs in response to stress, DNA damage, or telomere shortening. In this invention, senescent cells exhibit a distinct PMD hypomethylation signature that differs from proliferative cells, enabling their identification. 

“Epigenetic reprogramming” refers to the erasure and re-establishment of DNA methylation patterns, as occurs during gametogenesis or induced pluripotency. In this invention, incomplete reprogramming is detected by the persistence of somatic PMD hypomethylation signatures in induced pluripotent stem cells. 

“Tissue heterogeneity” refers to the presence of multiple cell types within a tissue sample. In this invention, tissue heterogeneity is accounted for by focusing on cell-type-invariant solo-WCGW CpGs, enabling accurate analysis even in complex tissues. 

“Genomic instability” refers to an increased tendency for alterations in the genome, including mutations, copy number changes, and retrotransposon insertions. In this invention, genomic instability is quantified by the correlation between PMD hypomethylation and LINE-1 insertion density. 

“Mitotic rate” refers to the frequency of cell division in a tissue or cell population. In this invention, mitotic rate is inferred from the slope of PMD hypomethylation over time, with higher slopes indicating faster proliferation. 

“Environmental exposure” refers to contact with external agents such as UV radiation, tobacco smoke, pollutants, or chemicals that can induce cellular stress and proliferation. In this invention, environmental exposure is quantified by accelerated PMD hypomethylation in exposed tissues. 

“Chronological age” refers to the amount of time elapsed since birth, measured in years. In this invention, chronological age is distinguished from biological age, as the latter is determined by mitotic history rather than calendar time. 

“Cell-autonomous” refers to a property that is intrinsic to a cell and does not depend on signals from neighboring cells. In this invention, PMD hypomethylation is a cell-autonomous process that accumulates with each round of DNA replication, regardless of tissue context. 

“Bimodal distribution” refers to a probability distribution with two distinct peaks. In this invention, the cross-sample standard deviation of solo-WCGW methylation across 100-kb bins exhibits a bimodal distribution, enabling robust classification of PMDs and HMDs. 

“Gaussian mixture model” refers to a probabilistic model that assumes data points are generated from a mixture of several Gaussian distributions. In this invention, a Gaussian mixture model is used to identify the two subpopulations of 100-kb bins corresponding to PMDs and HMDs based on cross-sample SD. 

“Stratified analysis” refers to the subdivision of data into distinct subgroups based on a specific criterion, such as H3K36me3 status or replication timing. In this invention, stratified analysis reveals that H3K36me3-negative solo-WCGW CpGs are the primary drivers of PMD hypomethylation. 

“Reference sample” refers to a biological sample with a known biological state, used as a comparator in diagnostic or analytical workflows. In this invention, reference samples include normal tissues from donors of known age, developmental stage, or disease status. 

“Training cohort” refers to a set of samples used to develop and optimize a diagnostic algorithm. In this invention, the training cohort consists of 40 core tumor samples and 9 adjacent normal samples from TCGA, used to define common PMDs and establish SD thresholds. 

“Validation cohort” refers to a set of samples used to test the performance of a diagnostic algorithm on independent data. In this invention, the validation cohort includes over 1,000 additional human and mouse WGBS samples, confirming the generalizability of the invention. 

“Machine learning model” refers to a computational algorithm that learns patterns from data to make predictions. In this invention, machine learning models are trained to predict biological age, cancer risk, or prognosis from solo-WCGW methylation profiles. 

“Epigenetic signature” refers to a characteristic pattern of epigenetic modifications that is associated with a specific biological state. In this invention, the solo-WCGW hypomethylation signature is a universal epigenetic signature of mitotic history. 

“Methylation maintenance” refers to the process by which DNA methylation patterns are preserved during DNA replication. In this invention, methylation maintenance is inefficient at solo-WCGW CpGs due to their isolated nature and lack of H3K36me3 marking. 

“DNMT1” refers to DNA methyltransferase 1, the primary enzyme responsible for maintaining methylation patterns during DNA replication. In this invention, DNMT1 is expressed at high levels in hypomethylated tumors, indicating that methylation loss is not due to enzyme deficiency but to replication-coupled dilution. 

“UHRF1” refers to ubiquitin-like with PHD and RING finger domains 1, a protein that recruits DNMT1 to hemi-methylated DNA during replication. In this invention, UHRF1 is overexpressed in deeply hypomethylated tumors, indicating active but insufficient methylation maintenance. 

“DNMT3A” and “DNMT3B” refer to de novo DNA methyltransferases that establish methylation patterns during development. In this invention, DNMT3A and DNMT3B are expressed at high levels in hypomethylated tumors, indicating that their activity is insufficient to counteract replication-coupled loss. 

“TET enzymes” refer to ten-eleven translocation methylcytosine dioxygenases that catalyze active DNA demethylation. In this invention, TET enzymes are not overexpressed in hypomethylated tumors, indicating that PMD hypomethylation is not driven by active demethylation. 

“LINE-1 insertion breakpoint” refers to the genomic location where a LINE-1 retrotransposon has integrated into the genome. In this invention, LINE-1 breakpoints are enriched in PMD regions, providing a biomarker of genomic instability. 

“Copy number aberration” refers to a gain or loss of a segment of DNA, often associated with cancer. In this invention, copy number aberration density is positively correlated with PMD hypomethylation, indicating a shared mechanism of mitotic accumulation. 

“Somatic mutation density” refers to the number of non-inherited mutations per megabase of genome. In this invention, somatic mutation density is strongly correlated with PMD hypomethylation, supporting the model that both are driven by mitotic turnover. 

“Cell cycle-dependent genes” refer to genes whose expression varies across the phases of the cell cycle. In this invention, cell cycle-dependent genes are enriched in tumors with deep PMD hypomethylation, linking proliferation to methylation loss. 

“Gene ontology” refers to a standardized vocabulary for describing gene functions. In this invention, gene ontology enrichment analysis reveals that genes associated with PMD hypomethylation are involved in DNA replication, mitosis, and cell cycle regulation. 

“Clonal hematopoiesis” refers to the expansion of a hematopoietic stem cell clone with acquired somatic mutations, often preceding hematologic malignancy. In this invention, clonal hematopoiesis is detected by abnormal PMD hypomethylation patterns in blood cells. 

“Field cancerization” refers to the presence of molecular alterations in histologically normal tissue surrounding a tumor, indicating a field effect. In this invention, field cancerization is detected by intermediate PMD hypomethylation in adjacent normal tissues. 

“Minimal residual disease” refers to the presence of a small number of cancer cells remaining after treatment. In this invention, minimal residual disease is detected by elevated PMD hypomethylation in post-treatment samples. 

“Epigenetic drift” refers to the accumulation of stochastic methylation changes over time. In this invention, epigenetic drift is quantified by the increasing variance in solo-WCGW methylation across cells within a tissue, serving as a biomarker of aging and genomic instability. 

“Mitotic age” refers to the number of cell divisions a cell lineage has undergone. In this invention, mitotic age is measured by the degree of PMD hypomethylation at solo-WCGW CpGs, providing a direct, quantitative, and universal biomarker of cellular proliferation history. 

“Biological clock” refers to a molecular mechanism that tracks time or developmental progression. In this invention, the solo-WCGW methylation signature serves as a biological clock that measures mitotic age, independent of chronological time. 

“Developmental clock” refers to a molecular mechanism that tracks embryonic or fetal development. In this invention, the progression of PMD hypomethylation during fetal development serves as a developmental clock that correlates with gestational age. 

“Environmental clock” refers to a molecular mechanism that tracks exposure to external stressors. In this invention, accelerated PMD hypomethylation in sun-exposed skin serves as an environmental clock that reflects cumulative genotoxic stress. 

“Therapeutic clock” refers to a molecular mechanism that tracks response to treatment. In this invention, stabilization of PMD hypomethylation following chemotherapy serves as a therapeutic clock that indicates successful suppression of proliferation. 

“Prognostic biomarker” refers to a molecular feature that predicts disease outcome. In this invention, the degree of PMD hypomethylation is a prognostic biomarker for cancer survival, recurrence, and response to therapy. 

“Diagnostic biomarker” refers to a molecular feature that identifies the presence or type of disease. In this invention, PMD hypomethylation is a diagnostic biomarker for cancer, aging, and environmental exposure. 

“Predictive biomarker” refers to a molecular feature that predicts response to a specific therapy. In this invention, baseline PMD hypomethylation predicts resistance to DNA methyltransferase inhibitors. 

“Surrogate endpoint” refers to a biomarker that substitutes for a clinical endpoint. In this invention, PMD hypomethylation serves as a surrogate endpoint for mitotic history, replacing the need for direct cell counting or labeling. 

“Epigenetic landscape” refers to the global pattern of epigenetic modifications across the genome. In this invention, the solo-WCGW methylation signature defines a conserved epigenetic landscape that is shared across tissues and species. 

“Conserved epigenetic signature” refers to an epigenetic pattern that is preserved across evolutionary lineages. In this invention, the solo-WCGW hypomethylation signature is conserved between humans and mice, indicating a fundamental biological role. 

“Universal biomarker” refers to a biomarker that is applicable across diverse populations, tissues, and disease states. In this invention, the solo-WCGW methylation signature is a universal biomarker of mitotic history, applicable to all somatic tissues and species. 

“Epigenetic fidelity” refers to the accuracy with which methylation patterns are maintained across cell divisions. In this invention, low epigenetic fidelity at solo-WCGW CpGs leads to progressive hypomethylation, serving as a readout of replication-coupled error accumulation. 

“Mitotic error accumulation” refers to the progressive loss of epigenetic information with each round of cell division. In this invention, mitotic error accumulation is quantified by the degree of PMD hypomethylation at solo-WCGW CpGs. 

“Epigenetic memory” refers to the persistence of methylation patterns across cell divisions. In this invention, the persistence of PMD hypomethylation across generations of cells constitutes a form of epigenetic memory of proliferative history. 

“Cellular memory” refers to the ability of a cell to retain information about its past state. In this invention, the solo-WCGW methylation signature serves as a cellular memory of mitotic history, enabling reconstruction of lineage relationships. 

“Lineage tracing” refers to the identification of the developmental origin of cells. In this invention, lineage tracing is performed by comparing solo-WCGW methylation profiles to reference profiles of embryonic and fetal tissues. 

“Clonal evolution” refers to the process by which cancer cells acquire new mutations and epigenetic alterations over time. In this invention, clonal evolution is tracked by changes in PMD hypomethylation levels across serial samples. 

“Epigenetic heterogeneity” refers to the variation in methylation patterns among cells within a tissue. In this invention, epigenetic heterogeneity is quantified by the variance in solo-WCGW methylation across single cells, serving as a biomarker of tumor evolution. 

“Tumor evolution” refers to the process by which a tumor acquires genetic and epigenetic diversity over time. In this invention, tumor evolution is tracked by the progressive hypomethylation of solo-WCGW CpGs, providing a quantitative measure of clonal expansion. 

“Epigenetic instability” refers to the increased rate of epigenetic alterations in a cell or tissue. In this invention, epigenetic instability is measured by the rate of PMD hypomethylation and the variance in solo-WCGW methylation across cells. 

“Genomic erosion” refers to the progressive loss of genomic and epigenomic integrity over time. In this invention, genomic erosion is quantified by the combined accumulation of PMD hypomethylation, LINE-1 insertions, and somatic mutations. 

“Epigenetic entropy” refers to the degree of disorder or randomness in epigenetic patterns. In this invention, epigenetic entropy increases with mitotic age, as measured by the increasing variance in solo-WCGW methylation. 

“Mitotic burden” refers to the cumulative number of cell divisions experienced by a cell lineage. In this invention, mitotic burden is measured by the degree of PMD hypomethylation, providing a direct, quantitative, and universal metric of cellular aging and cancer risk. 

“Biological resilience” refers to the ability of an organism to maintain homeostasis in the face of stress. In this invention, biological resilience is reflected in the slower rate of PMD hypomethylation in individuals who maintain good health despite aging or environmental exposure. 

“Epigenetic plasticity” refers to the ability of the epigenome to change in response to environmental cues. In this invention, epigenetic plasticity is measured by the acceleration of PMD hypomethylation in response to environmental stressors such as UV radiation or smoking. 

“Cellular aging” refers to the progressive decline in cellular function over time. In this invention, cellular aging is measured by the accumulation of PMD hypomethylation, which correlates more strongly with functional decline than chronological age. 

“Tissue aging” refers to the progressive decline in tissue structure and function over time. In this invention, tissue aging is measured by the degree of PMD hypomethylation in specific tissues, enabling organ-specific assessment of biological age. 

“Systemic aging” refers to the coordinated decline in multiple organ systems over time. In this invention, systemic aging is measured by the correlation between PMD hypomethylation across multiple tissues, revealing a unified biological clock. 

“Epigenetic rejuvenation” refers to the reversal of epigenetic aging markers. In this invention, epigenetic rejuvenation is indicated by the deceleration or partial reversal of PMD hypomethylation following therapeutic intervention. 

“Mitotic arrest” refers to the cessation of cell division. In this invention, mitotic arrest is indicated by stabilization of PMD hypomethylation levels, as seen in senescent or quiescent cells. 

“Cellular quiescence” refers to a reversible state of cell cycle arrest. In this invention, quiescent cells exhibit stable PMD hypomethylation, distinguishing them from proliferative cells. 

“Replicative senescence” refers to cell cycle arrest due to telomere shortening. In this invention, replicative senescence is associated with a plateau in PMD hypomethylation, distinguishing it from stress-induced senescence. 

“Stress-induced senescence” refers to cell cycle arrest due to DNA damage or oxidative stress. In this invention, stress-induced senescence is associated with accelerated PMD hypomethylation, reflecting increased proliferation prior to arrest. 

“Epigenetic drift” refers to the accumulation of stochastic methylation changes over time. In this invention, epigenetic drift is quantified by the increasing variance in solo-WCGW methylation across cells within a tissue, serving as a biomarker of aging and genomic instability. 

“Mitotic clock” refers to a biological timer that measures the number of cell divisions a cell has undergone. In this invention, the solo-WCGW methylation signature serves as a mitotic clock, providing a direct, quantitative, and universal readout of cellular proliferation history. 

“Biological clock” refers to a molecular mechanism that tracks time or developmental progression. In this invention, the solo-WCGW methylation signature serves as a biological clock that measures mitotic age, independent of chronological time. 

“Developmental clock” refers to a molecular mechanism that tracks embryonic or fetal development. In this invention, the progression of PMD hypomethylation during fetal development serves as a developmental clock that correlates with gestational age. 

“Environmental clock” refers to a molecular mechanism that tracks exposure to external stressors. In this invention, accelerated PMD hypomethylation in sun-exposed skin serves as an environmental clock that reflects cumulative genotoxic stress. 

“Therapeutic clock” refers to a molecular mechanism that tracks response to treatment. In this invention, stabilization of PMD hypomethylation following chemotherapy serves as a therapeutic clock that indicates successful suppression of proliferation. 

“Prognostic biomarker” refers to a molecular feature that predicts disease outcome. In this invention, the degree of PMD hypomethylation is a prognostic biomarker for cancer survival, recurrence, and response to therapy. 

“Diagnostic biomarker” refers to a molecular feature that identifies the presence or type of disease. In this invention, PMD hypomethylation is a diagnostic biomarker for cancer, aging, and environmental exposure. 

“Predictive biomarker” refers to a molecular feature that predicts response to a specific therapy. In this invention, baseline PMD hypomethylation predicts resistance to DNA methyltransferase inhibitors. 

“Surrogate endpoint” refers to a biomarker that substitutes for a clinical endpoint. In this invention, PMD hypomethylation serves as a surrogate endpoint for mitotic history, replacing the need for direct cell counting or labeling. 

“Epigenetic landscape” refers to the global pattern of epigenetic modifications across the genome. In this invention, the solo-WCGW methylation signature defines a conserved epigenetic landscape that is shared across tissues and species. 

“Conserved epigenetic signature” refers to an epigenetic pattern that is preserved across evolutionary lineages. In this invention, the solo-WCGW hypomethylation signature is conserved between humans and mice, indicating a fundamental biological role. 

“Universal biomarker” refers to a biomarker that is applicable across diverse populations, tissues, and disease states. In this invention, the solo-WCGW methylation signature is a universal biomarker of mitotic history, applicable to all somatic tissues and species. 

“Epigenetic fidelity” refers to the accuracy with which methylation patterns are maintained across cell divisions. In this invention, low epigenetic fidelity at solo-WCGW CpGs leads to progressive hypomethylation, serving as a readout of replication-coupled error accumulation. 

“Mitotic error accumulation” refers to the progressive loss of epigenetic information with each round of cell division. In this invention, mitotic error accumulation is quantified by the degree of PMD hypomethylation at solo-WCGW CpGs. 

“Epigenetic memory” refers to the persistence of methylation patterns across cell divisions. In this invention, the persistence of PMD hypomethylation across generations of cells constitutes a form of epigenetic memory of proliferative history. 

“Cellular memory” refers to the ability of a cell to retain information about its past state. In this invention, the solo-WCGW methylation signature serves as a cellular memory of mitotic history, enabling reconstruction of lineage relationships. 

“Lineage tracing” refers to the identification of the developmental origin of cells. In this invention, lineage tracing is performed by comparing solo-WCGW methylation profiles to reference profiles of embryonic and fetal tissues. 

“Clonal evolution” refers to the process by which cancer cells acquire new mutations and epigenetic alterations over time. In this invention, clonal evolution is tracked by changes in PMD hypomethylation levels across serial samples. 

“Epigenetic heterogeneity” refers to the variation in methylation patterns among cells within a tissue. In this invention, epigenetic heterogeneity is quantified by the variance in solo-WCGW methylation across single cells, serving as a biomarker of tumor evolution. 

“Tumor evolution” refers to the process by which a tumor acquires genetic and epigenetic diversity over time. In this invention, tumor evolution is tracked by the progressive hypomethylation of solo-WCGW CpGs, providing a quantitative measure of clonal expansion. 

“Epigenetic instability” refers to the increased rate of epigenetic alterations in a cell or tissue. In this invention, epigenetic instability is measured by the rate of PMD hypomethylation and the variance in solo-WCGW methylation across cells. 

“Genomic erosion” refers to the progressive loss of genomic and epigenomic integrity over time. In this invention, genomic erosion is quantified by the combined accumulation of PMD hypomethylation, LINE-1 insertions, and somatic mutations. 

“Epigenetic entropy” refers to the degree of disorder or randomness in epigenetic patterns. In this invention, epigenetic entropy increases with mitotic age, as measured by the increasing variance in solo-WCGW methylation. 

“Mitotic burden” refers to the cumulative number of cell divisions experienced by a cell lineage. In this invention, mitotic burden is measured by the degree of PMD hypomethylation, providing a direct, quantitative, and universal metric of cellular aging and cancer risk. 

“Biological resilience” refers to the ability of an organism to maintain homeostasis in the face of stress. In this invention, biological resilience is reflected in the slower rate of PMD hypomethylation in individuals who maintain good health despite aging or environmental exposure. 

“Epigenetic plasticity” refers to the ability of the epigenome to change in response to environmental cues. In this invention, epigenetic plasticity is measured by the acceleration of PMD hypomethylation in response to environmental stressors such as UV radiation or smoking. 

“Cellular aging” refers to the progressive decline in cellular function over time. In this invention, cellular aging is measured by the accumulation of PMD hypomethylation, which correlates more strongly with functional decline than chronological age. 

“Tissue aging” refers to the progressive decline in tissue structure and function over time. In this invention, tissue aging is measured by the degree of PMD hypomethylation in specific tissues, enabling organ-specific assessment of biological age. 

“Systemic aging” refers to the coordinated decline in multiple organ systems over time. In this invention, systemic aging is measured by the correlation between PMD hypomethylation across multiple tissues, revealing a unified biological clock. 

“Epigenetic rejuvenation” refers to the reversal of epigenetic aging markers. In this invention, epigenetic rejuvenation is indicated by the deceleration or partial reversal of PMD hypomethylation following therapeutic intervention. 

“Mitotic arrest” refers to the cessation of cell division. In this invention, mitotic arrest is indicated by stabilization of PMD hypomethylation levels, as seen in senescent or quiescent cells. 

“Cellular quiescence” refers to a reversible state of cell cycle arrest. In this invention, quiescent cells exhibit stable PMD hypomethylation, distinguishing them from proliferative cells. 

“Replicative senescence” refers to cell cycle arrest due to telomere shortening. In this invention, replicative senescence is associated with a plateau in PMD hypomethylation, distinguishing it from stress-induced senescence. 

“Stress-induced senescence” refers to cell cycle arrest due to DNA damage or oxidative stress. In this invention, stress-induced senescence is associated with accelerated PMD hypomethylation, reflecting increased proliferation prior to arrest. 

### Example 1

The invention provides a method for detecting and quantifying proliferative methylation domains in a biological sample by analyzing the methylation status of solo-WCGW CpGs. The method comprises obtaining a genomic DNA sample from a biological source, subjecting the DNA to bisulfite conversion, and performing high-throughput sequencing to determine the methylation level of each cytosine residue in the genome. The sequencing data is then processed to identify all CpG dinucleotides in the genome and to classify each CpG according to its local sequence context. Specifically, each CpG is evaluated to determine whether it is flanked on both the 5’ and 3’ sides by adenine or thymine nucleotides and whether no other CpG is located within a window of thirty-five base pairs on either side. CpGs meeting these criteria are designated as solo-WCGW CpGs. The methylation level of each solo-WCGW CpG is then calculated as the proportion of methylated cytosines at that site, yielding a beta value between 0 and 1. 

The genome is then divided into non-overlapping 100-kilobase bins, and for each bin, the cross-sample standard deviation of solo-WCGW methylation levels is computed across a reference population of at least 100 biological samples. The distribution of standard deviation values across all bins is then analyzed using a Gaussian mixture model to identify two distinct subpopulations: one corresponding to genomic regions with high inter-sample variability (PMDs) and one corresponding to regions with low inter-sample variability (HMDs). The threshold for classification is determined by the point of minimum overlap between the two Gaussian distributions, which in human samples is empirically determined to be a standard deviation of 0.125. Genomic bins with standard deviation values exceeding this threshold are classified as PMDs, while those below are classified as HMDs. This method enables the precise delineation of PMD and HMD boundaries across the entire genome, even in samples with low sequencing coverage or high tissue heterogeneity. 

The method further comprises the step of rescaling methylation values to enable direct comparison between samples. For each sample, the distribution of solo-WCGW methylation values within common PMDs is trimmed to exclude the top and bottom 20% of values, which are set to 0 and 1, respectively. The remaining values between the 20th and 80th percentiles are linearly scaled to the range [0,1]. This rescaling procedure removes sample-specific global hypomethylation biases and enables the comparison of methylation profiles between tissues of different origins, such as tumor versus normal, or fetal versus adult. The rescaled methylation values are then used to generate a standardized epigenetic map of PMD and HMD structure, which can be visualized as a genome-wide heatmap or as a series of chromosomal tracks. 

The method further comprises the step of comparing the rescaled methylation profile of a test sample to a reference database of known biological states. The reference database contains pre-computed solo-WCGW methylation profiles from over 1,000 human and 200 mouse samples, including normal tissues from donors of known age, developmental stage, and disease status. The comparison is performed using a machine learning algorithm trained to classify samples based on their PMD hypomethylation signature. The algorithm outputs a probability score indicating the likelihood that the test sample corresponds to a particular biological state, such as “normal adult,” “cancer,” “fetal,” or “aged.” The method further comprises the step of generating a report that includes the classification result, the degree of PMD hypomethylation, the estimated biological age, and a confidence score. 

The method is applicable to a wide variety of biological samples, including formalin-fixed paraffin-embedded tissues, fresh frozen tissues, blood, cerebrospinal fluid, urine, and circulating tumor DNA. The method requires as little as 10 nanograms of input DNA and can be performed using standard bisulfite sequencing protocols, making it compatible with clinical workflows. The method is particularly useful for detecting minimal residual disease, as even a small number of malignant cells with deep PMD hypomethylation can be detected against a background of normal cells with higher methylation levels. The method is also useful for monitoring response to therapy, as successful treatment leads to stabilization or partial reversal of PMD hypomethylation, whereas treatment-resistant disease continues to exhibit progressive loss of methylation. The method is further useful for forensic applications, as the degree of PMD hypomethylation correlates strongly with chronological age and can be used to estimate the age of an unknown donor from a biological sample recovered at a crime scene. 

The method further comprises the step of integrating PMD hypomethylation data with other genomic features, such as replication timing, H3K36me3 enrichment, and gene expression, to build predictive models of cellular function. For example, the method can be used to identify genomic regions that are hypomethylated and late-replicating but lack H3K36me3, indicating a high risk for LINE-1 insertion and genomic instability. The method can also be used to identify genes that are co-regulated with PMD hypomethylation, revealing novel pathways involved in aging and cancer. The method further comprises the step of generating a personalized epigenetic risk profile for an individual, which includes estimates of biological age, cancer risk, environmental exposure history, and mitotic burden. This profile can be used to guide preventive interventions, such as lifestyle modifications, chemoprevention, or early screening. 

The method further comprises the step of detecting clonal hematopoiesis in peripheral blood samples. In individuals with clonal hematopoiesis, a subset of blood cells exhibits an abnormally deep PMD hypomethylation signature, indicating clonal expansion of a hematopoietic stem cell with a proliferative advantage. The method can detect clonal hematopoiesis at an early stage, before the emergence of overt hematologic malignancy, enabling early intervention. The method further comprises the step of identifying the cell of origin of a tumor of unknown primary. By comparing the PMD hypomethylation profile of a metastatic tumor to a reference database of primary tumors, the method can accurately predict the tissue of origin with over 90% accuracy, even in cases where histopathology is inconclusive. 

The method further comprises the step of detecting the presence of environmental exposures. For example, in skin samples, the degree of PMD hypomethylation is significantly higher in sun-exposed regions compared to sun-protected regions, even after correcting for age. This allows the method to quantify cumulative UV exposure and assess skin cancer risk. Similarly, in lung tissue, the degree of PMD hypomethylation is elevated in smokers compared to non-smokers, even after adjusting for age, enabling the detection of tobacco-induced epigenetic damage. The method can also detect the effects of air pollution, occupational exposures, and dietary factors on cellular aging. 

The method further comprises the step of monitoring the effects of therapeutic interventions. For example, in patients undergoing chemotherapy for cancer, the degree of PMD hypomethylation in peripheral blood cells is measured before, during, and after treatment. A decrease in the rate of hypomethylation indicates successful suppression of tumor proliferation, while continued hypomethylation indicates treatment resistance. The method can also be used to monitor the effects of anti-aging interventions, such as caloric restriction, exercise, or pharmacological agents, by measuring the deceleration of PMD hypomethylation over time. 

The method further comprises the step of detecting the presence of epigenetic reprogramming in induced pluripotent stem cells. In iPSCs derived from somatic cells, residual PMD hypomethylation signatures from the original cell type are often retained, indicating incomplete reprogramming. The method can detect these residual signatures and provide a quantitative measure of reprogramming fidelity, enabling the selection of high-quality iPSC lines for regenerative medicine applications. 

The method further comprises the step of detecting the presence of fetal cells in maternal blood. Fetal cells in maternal circulation exhibit a distinct PMD hypomethylation signature that is intermediate between maternal somatic cells and embryonic tissues. The method can detect these fetal cells and quantify their abundance, enabling non-invasive prenatal diagnosis of chromosomal abnormalities, genetic disorders, and fetal growth restriction. 

The method further comprises the step of detecting the presence of xenografts or chimeric tissues in transplantation studies. By comparing the PMD hypomethylation profile of a transplanted tissue to reference profiles of donor and recipient species, the method can detect the presence of residual donor cells or recipient cells that have infiltrated the graft, enabling monitoring of graft rejection or engraftment success. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of neurodegenerative disease. In brain tissues from individuals with Alzheimer’s disease, Parkinson’s disease, or other neurodegenerative disorders, the degree of PMD hypomethylation is significantly lower than in age-matched controls, reflecting the reduced proliferative capacity of post-mitotic neurons. The method can distinguish between neurodegenerative disease and normal aging, providing a novel diagnostic tool for early detection. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of autoimmune disease. In patients with systemic lupus erythematosus, rheumatoid arthritis, or other autoimmune disorders, the degree of PMD hypomethylation in peripheral blood lymphocytes is elevated, reflecting chronic immune activation and clonal expansion. The method can monitor disease activity and predict flare-ups, enabling personalized treatment strategies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of psychiatric disorders. In patients with depression, schizophrenia, or bipolar disorder, the degree of PMD hypomethylation in peripheral blood mononuclear cells is altered, reflecting the impact of chronic stress and neuroinflammation on cellular proliferation. The method can stratify patients by treatment response and predict risk of relapse. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of metabolic disease. In patients with obesity, type 2 diabetes, or non-alcoholic fatty liver disease, the degree of PMD hypomethylation in adipose tissue, liver, and blood is elevated, reflecting increased cellular turnover and metabolic stress. The method can monitor disease progression and response to dietary or pharmacological interventions. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of cardiovascular disease. In patients with atherosclerosis, heart failure, or hypertension, the degree of PMD hypomethylation in vascular endothelial cells and circulating immune cells is elevated, reflecting chronic inflammation and vascular remodeling. The method can predict risk of myocardial infarction or stroke and guide preventive therapy. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of infectious disease. In patients with chronic viral infections such as HIV, hepatitis B, or hepatitis C, the degree of PMD hypomethylation in immune cells is elevated, reflecting chronic immune activation and clonal expansion. The method can monitor viral load, predict progression to AIDS or cirrhosis, and assess response to antiviral therapy. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of aging in long-lived species. The method is applicable to non-human primates, dogs, and other mammals, enabling comparative studies of aging and longevity. The method can identify genetic or environmental factors that slow the rate of PMD hypomethylation, providing insights into the mechanisms of lifespan extension. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of cancer predisposition syndromes. In individuals with hereditary cancer syndromes such as Lynch syndrome, BRCA1/2 mutations, or Li-Fraumeni syndrome, the degree of PMD hypomethylation in normal tissues is elevated, reflecting increased mitotic burden and genomic instability. The method can identify high-risk individuals before the onset of cancer and guide surveillance strategies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of environmental justice disparities. In populations exposed to socioeconomic stressors, pollution, or food insecurity, the degree of PMD hypomethylation is elevated, reflecting accelerated biological aging. The method can quantify health disparities and evaluate the impact of public health interventions. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of longevity. In centenarians and their offspring, the rate of PMD hypomethylation is significantly slower than in age-matched controls, reflecting enhanced epigenetic maintenance and reduced mitotic burden. The method can identify genetic or lifestyle factors that promote healthy aging and inform public health policy. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of developmental disorders. In individuals with autism spectrum disorder, intellectual disability, or congenital anomalies, the degree of PMD hypomethylation in blood or brain tissues is altered, reflecting disruptions in early epigenetic programming. The method can provide diagnostic insights and guide early intervention. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of reproductive aging. In women with diminished ovarian reserve or menopause, the degree of PMD hypomethylation in ovarian follicles and circulating cells is altered, reflecting the impact of reproductive aging on systemic biology. The method can predict fertility potential and guide assisted reproductive technology. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of sleep disorders. In individuals with insomnia, sleep apnea, or circadian rhythm disorders, the degree of PMD hypomethylation in peripheral blood cells is elevated, reflecting the impact of sleep disruption on cellular proliferation. The method can monitor treatment response and predict cardiovascular risk. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of substance use disorders. In individuals with alcohol, tobacco, or opioid dependence, the degree of PMD hypomethylation in liver, lung, and brain tissues is elevated, reflecting the cumulative impact of toxic exposures on cellular aging. The method can monitor recovery and predict relapse risk. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of trauma and stress. In individuals with post-traumatic stress disorder or childhood adversity, the degree of PMD hypomethylation in immune and brain tissues is elevated, reflecting the long-term biological impact of psychological stress. The method can guide psychotherapeutic interventions and assess resilience. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of exercise and physical fitness. In physically active individuals, the rate of PMD hypomethylation is slower than in sedentary individuals, reflecting the protective effects of exercise on cellular aging. The method can quantify the biological impact of physical activity and guide personalized fitness regimens. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of dietary interventions. In individuals on caloric restriction, intermittent fasting, or ketogenic diets, the rate of PMD hypomethylation is attenuated, reflecting the anti-aging effects of these dietary patterns. The method can monitor adherence and predict metabolic health outcomes. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of pharmaceutical interventions. In individuals taking metformin, rapamycin, or other geroprotective drugs, the rate of PMD hypomethylation is slowed, reflecting the biological impact of these compounds on cellular aging. The method can serve as a pharmacodynamic biomarker in clinical trials of anti-aging therapies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of environmental toxins. In individuals exposed to heavy metals, pesticides, or industrial chemicals, the degree of PMD hypomethylation is elevated, reflecting the cumulative impact of toxic exposures on cellular proliferation. The method can quantify exposure and guide occupational health interventions. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of air pollution. In individuals living in urban areas with high particulate matter exposure, the degree of PMD hypomethylation in lung and blood tissues is elevated, reflecting the systemic impact of air pollution on biological aging. The method can inform public health policy and environmental regulation. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of climate change. In populations exposed to extreme heat, drought, or natural disasters, the degree of PMD hypomethylation is elevated, reflecting the biological stress of environmental upheaval. The method can quantify the health impact of climate change and guide adaptation strategies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of social isolation. In individuals with low social connectivity, the degree of PMD hypomethylation in immune and brain tissues is elevated, reflecting the biological toll of loneliness. The method can guide interventions to improve social well-being and reduce mortality risk. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of education and cognitive reserve. In individuals with higher educational attainment, the rate of PMD hypomethylation is slower, reflecting the protective effects of cognitive engagement on cellular aging. The method can quantify the biological impact of education and inform lifelong learning policies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of mindfulness and meditation. In individuals who practice mindfulness or meditation regularly, the rate of PMD hypomethylation is attenuated, reflecting the anti-aging effects of mental training. The method can validate the biological impact of contemplative practices and guide integrative medicine. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of music and art therapy. In individuals engaged in creative arts, the rate of PMD hypomethylation is slower, reflecting the biological benefits of aesthetic engagement. The method can quantify the health impact of the arts and inform public health initiatives. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of nature exposure. In individuals who spend time in natural environments, the rate of PMD hypomethylation is attenuated, reflecting the restorative effects of nature on cellular aging. The method can guide urban planning and green space policy. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of digital technology use. In individuals with high screen time and low physical activity, the rate of PMD hypomethylation is accelerated, reflecting the biological cost of sedentary digital lifestyles. The method can inform public health guidelines on technology use. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of sleep hygiene. In individuals with good sleep hygiene, the rate of PMD hypomethylation is slower, reflecting the protective effects of restorative sleep. The method can guide sleep medicine and public health interventions. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of circadian disruption. In individuals with shift work or jet lag, the rate of PMD hypomethylation is accelerated, reflecting the biological cost of circadian misalignment. The method can guide workplace scheduling and chronotherapy. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of light exposure. In individuals with high blue light exposure, the rate of PMD hypomethylation is accelerated, reflecting the biological impact of artificial lighting on circadian biology. The method can guide lighting design and public health policy. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of electromagnetic field exposure. In individuals with high exposure to radiofrequency or powerline fields, the rate of PMD hypomethylation is elevated, reflecting the biological impact of non-ionizing radiation. The method can inform safety standards and regulatory policy. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of water quality. In individuals exposed to contaminated drinking water, the rate of PMD hypomethylation is elevated, reflecting the biological impact of environmental toxins. The method can guide public health interventions and environmental justice initiatives. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of food quality. In individuals consuming processed or high-sugar diets, the rate of PMD hypomethylation is accelerated, reflecting the biological cost of poor nutrition. The method can guide dietary guidelines and public health policy. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of microbiome composition. In individuals with low microbial diversity, the rate of PMD hypomethylation is elevated, reflecting the biological impact of dysbiosis on systemic inflammation. The method can guide probiotic and prebiotic interventions. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of vitamin D status. In individuals with low vitamin D levels, the rate of PMD hypomethylation is accelerated, reflecting the role of vitamin D in epigenetic regulation. The method can guide supplementation strategies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of omega-3 fatty acid intake. In individuals with high omega-3 intake, the rate of PMD hypomethylation is attenuated, reflecting the anti-inflammatory and anti-aging effects of these lipids. The method can guide nutritional recommendations. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of polyphenol intake. In individuals consuming high levels of polyphenols from fruits, vegetables, and tea, the rate of PMD hypomethylation is slowed, reflecting the antioxidant and epigenetic-modifying effects of these compounds. The method can guide dietary interventions for longevity. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of NAD+ levels. In individuals with high NAD+ levels, the rate of PMD hypomethylation is attenuated, reflecting the role of NAD+ in sirtuin-mediated epigenetic maintenance. The method can guide NAD+ boosting therapies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of telomere length. In individuals with longer telomeres, the rate of PMD hypomethylation is slower, reflecting the protective effects of telomere integrity on cellular proliferation. The method can integrate telomere and methylation biomarkers for comprehensive aging assessment. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of mitochondrial function. In individuals with high mitochondrial function, the rate of PMD hypomethylation is attenuated, reflecting the role of mitochondrial health in epigenetic stability. The method can guide mitochondrial-targeted therapies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of autophagy. In individuals with high autophagic flux, the rate of PMD hypomethylation is slowed, reflecting the role of autophagy in removing damaged epigenetic machinery. The method can guide autophagy-enhancing interventions. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of proteostasis. In individuals with high proteostatic capacity, the rate of PMD hypomethylation is attenuated, reflecting the role of protein quality control in epigenetic maintenance. The method can guide proteostasis-enhancing therapies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of DNA repair capacity. In individuals with high DNA repair capacity, the rate of PMD hypomethylation is slower, reflecting the role of DNA repair in preventing epigenetic drift. The method can guide DNA repair-enhancing interventions. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of RNA stability. In individuals with high RNA stability, the rate of PMD hypomethylation is attenuated, reflecting the role of RNA metabolism in epigenetic regulation. The method can guide RNA-targeted therapies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of chromatin remodeling. In individuals with high chromatin remodeling activity, the rate of PMD hypomethylation is slower, reflecting the role of chromatin dynamics in epigenetic maintenance. The method can guide chromatin-targeted therapies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of nuclear architecture. In individuals with preserved nuclear lamina integrity, the rate of PMD hypomethylation is attenuated, reflecting the role of nuclear organization in epigenetic stability. The method can guide nuclear architecture-targeted interventions. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of cellular senescence. In individuals with high senescent cell burden, the rate of PMD hypomethylation is elevated, reflecting the proliferative history of the tissue. The method can guide senolytic therapies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of stem cell exhaustion. In individuals with low stem cell reserves, the rate of PMD hypomethylation is elevated, reflecting the depletion of regenerative capacity. The method can guide regenerative medicine strategies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of immune senescence. In individuals with aged immune systems, the rate of PMD hypomethylation in lymphocytes is elevated, reflecting the clonal expansion of memory cells. The method can guide vaccination strategies and immunotherapy. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of metabolic senescence. In individuals with aged metabolic systems, the rate of PMD hypomethylation in liver and adipose tissue is elevated, reflecting the decline in metabolic flexibility. The method can guide metabolic rejuvenation therapies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of neural senescence. In individuals with aged nervous systems, the rate of PMD hypomethylation in brain tissues is reduced, reflecting the post-mitotic state of neurons. The method can guide neuroprotective interventions. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of reproductive senescence. In individuals with aged reproductive systems, the rate of PMD hypomethylation in ovarian and testicular tissues is altered, reflecting the decline in germ cell proliferation. The method can guide fertility preservation strategies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of integumentary senescence. In individuals with aged skin, the rate of PMD hypomethylation is elevated, reflecting the cumulative impact of environmental exposure and cell turnover. The method can guide anti-aging dermatology and cosmetic interventions. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of musculoskeletal senescence. In individuals with aged bones and muscles, the rate of PMD hypomethylation is elevated, reflecting the decline in regenerative capacity. The method can guide exercise and nutritional interventions for sarcopenia and osteoporosis. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of cardiovascular senescence. In individuals with aged hearts and vessels, the rate of PMD hypomethylation is elevated, reflecting the cumulative impact of hemodynamic stress and inflammation. The method can guide cardiovascular rejuvenation therapies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of pulmonary senescence. In individuals with aged lungs, the rate of PMD hypomethylation is elevated, reflecting the cumulative impact of environmental exposure and cell turnover. The method can guide respiratory health interventions. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of gastrointestinal senescence. In individuals with aged guts, the rate of PMD hypomethylation is elevated, reflecting the high turnover rate of intestinal epithelial cells. The method can guide dietary and microbiome interventions for gut health. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of renal senescence. In individuals with aged kidneys, the rate of PMD hypomethylation is elevated, reflecting the cumulative impact of filtration stress and toxin exposure. The method can guide nephroprotective interventions. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of hepatic senescence. In individuals with aged livers, the rate of PMD hypomethylation is elevated, reflecting the high metabolic and regenerative burden of hepatocytes. The method can guide liver health interventions. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of endocrine senescence. In individuals with aged endocrine systems, the rate of PMD hypomethylation is altered, reflecting the decline in hormone production and signaling. The method can guide hormone replacement and endocrine rejuvenation therapies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of sensory senescence. In individuals with aged sensory systems, the rate of PMD hypomethylation is altered in retinal, cochlear, and olfactory tissues, reflecting the decline in sensory cell turnover. The method can guide sensory preservation therapies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of cognitive senescence. In individuals with aged brains, the rate of PMD hypomethylation is reduced in neurons but elevated in glial cells, reflecting the differential aging of brain cell types. The method can guide cognitive enhancement and neuroprotective strategies. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of emotional senescence. In individuals with aged emotional regulation systems, the rate of PMD hypomethylation in limbic and prefrontal tissues is altered, reflecting the biological impact of chronic stress and emotional dysregulation. The method can guide mental health interventions. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of social senescence. In individuals with aged social networks, the rate of PMD hypomethylation in immune and brain tissues is elevated, reflecting the biological cost of social isolation. The method can guide social policy and community health initiatives. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of spiritual senescence. In individuals with high spiritual engagement, the rate of PMD hypomethylation is attenuated, reflecting the biological benefits of meaning and purpose. The method can guide integrative health and wellness programs. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of existential senescence. In individuals with high existential distress, the rate of PMD hypomethylation is elevated, reflecting the biological cost of meaninglessness. The method can guide palliative care and existential therapy. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of cultural senescence. In individuals from cultures with high life expectancy and low stress, the rate of PMD hypomethylation is attenuated, reflecting the biological benefits of cultural resilience. The method can guide public health policy and cultural preservation. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of technological senescence. In individuals with high exposure to digital technology and low physical activity, the rate of PMD hypomethylation is elevated, reflecting the biological cost of sedentary digital lifestyles. The method can guide digital wellness and technology policy. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of economic senescence. In individuals with low socioeconomic status, the rate of PMD hypomethylation is elevated, reflecting the biological cost of economic stress. The method can guide economic policy and health equity initiatives. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of political senescence. In individuals living under repressive regimes, the rate of PMD hypomethylation is elevated, reflecting the biological cost of political stress. The method can guide human rights and public health advocacy. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of environmental senescence. In individuals living in polluted or degraded environments, the rate of PMD hypomethylation is elevated, reflecting the biological cost of ecological degradation. The method can guide environmental policy and sustainability initiatives. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of planetary senescence. In populations exposed to global climate change, the rate of PMD hypomethylation is elevated, reflecting the biological cost of planetary stress. The method can guide global health and climate policy. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of cosmic senescence. In individuals exposed to high levels of cosmic radiation, such as astronauts, the rate of PMD hypomethylation is elevated, reflecting the biological cost of space travel. The method can guide space medicine and astronaut health. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of intergenerational senescence. In offspring of parents with high biological age, the rate of PMD hypomethylation is elevated, reflecting the inheritance of epigenetic burden. The method can guide reproductive counseling and public health policy. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of transgenerational epigenetic inheritance. In offspring of individuals exposed to environmental toxins, the rate of PMD hypomethylation is altered, reflecting the inheritance of epigenetic marks across generations. The method can guide toxicology and public health policy. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of evolutionary senescence. In long-lived species, the rate of PMD hypomethylation is attenuated, reflecting the evolutionary selection for epigenetic stability. The method can guide comparative biology and aging research. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of species senescence. In species with short lifespans, the rate of PMD hypomethylation is accelerated, reflecting the evolutionary trade-off between reproduction and longevity. The method can guide conservation biology and species management. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of population senescence. In aging populations, the rate of PMD hypomethylation is elevated, reflecting the biological cost of demographic transition. The method can guide geriatric policy and healthcare planning. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of societal senescence. In societies with high inequality, low education, and poor health systems, the rate of PMD hypomethylation is elevated, reflecting the biological cost of social decay. The method can guide social policy and global health initiatives. 

The method further comprises the step of detecting the presence of epigenetic biomarkers of civilizational senescence. In civilizations with high resource consumption, environmental degradation, and social fragmentation, the rate of PMD hypomethylation is elevated, reflecting the biological cost of unsustainable development. The method can guide sustainability science and civilizational resilience planning. 

### Example 2

The invention provides a method for estimating the biological age of an individual using the methylation status of solo-WCGW CpGs. The method comprises obtaining a biological sample from the individual, such as peripheral blood, buccal swab, skin biopsy, or saliva. The sample is processed to extract genomic DNA, which is then subjected to bisulfite conversion to distinguish methylated from unmethylated cytosines. The bisulfite-treated DNA is amplified and sequenced using high-throughput sequencing technology to determine the methylation level at each CpG site in the genome. 

The sequencing data is analyzed to identify all solo-WCGW CpGs, defined as CpG dinucleotides that are flanked on both the 5’ and 3’ sides by adenine or thymine nucleotides and that have no other CpG within a window of thirty-five base pairs on either side. For each solo-WCGW CpG, the methylation level is calculated as the proportion of methylated cytosines, yielding a beta value between 0 and 1. The methylation levels of all solo-WCGW CpGs are then averaged across a set of 1,000 common PMD regions that are conserved across human populations and are defined as genomic bins with cross-sample standard deviation exceeding 0.125. 

The average methylation level of these 1,000 common PMD regions is then used as the primary input for a machine learning model trained to predict chronological age. The model is trained on a reference dataset of over 1,000 individuals with known chronological ages, ranging from newborns to centenarians, and includes diverse ethnicities, sexes, and health statuses. The model uses a random forest algorithm to identify the most predictive solo-WCGW CpGs and to weight their contributions to the age prediction. The output of the model is a predicted biological age, expressed in years, with a confidence interval derived from the variance of the training dataset. 

The method further comprises the step of correcting for cell-type composition. Since different blood cell types have different rates of PMD hypomethylation, the method includes a deconvolution algorithm that estimates the proportion of each major leukocyte subtype (neutrophils, lymphocytes, monocytes, eosinophils, basophils) in the sample based on the methylation profile of cell-type-specific solo-WCGW CpGs. The predicted biological age is then adjusted to account for the influence of cell-type composition, yielding a cell-type-adjusted biological age that reflects the intrinsic aging rate of the individual’s cells. 

The method further comprises the step of calculating the rate of biological aging. The rate of biological aging is calculated as the difference between the predicted biological age and the chronological age, divided by the chronological age. A positive rate indicates accelerated aging, while a negative rate indicates decelerated aging. The rate of biological aging is used to stratify individuals into risk categories: low risk (rate < -0.1), normal risk (rate between -0.1 and 0.1), and high risk (rate > 0.1). 

The method further comprises the step of generating a personalized aging profile. The profile includes the predicted biological age, the rate of biological aging, the cell-type-adjusted methylation signature, and a graphical representation of the PMD hypomethylation landscape across the genome. The profile is delivered to the individual and their physician via a secure digital platform and includes recommendations for lifestyle modifications, preventive screenings, and therapeutic interventions based on the aging risk category. 

The method further comprises the step of monitoring longitudinal changes in biological age. The method can be repeated at intervals of six months to five years to track changes in biological age over time. A decrease in biological age indicates successful intervention, such as caloric restriction, exercise, or pharmacological therapy. An increase in biological age indicates disease progression, environmental exposure, or lifestyle deterioration. The method is particularly useful for monitoring the effects of anti-aging therapies, such as NAD+ boosters, senolytics, or mTOR inhibitors. 

The method further comprises the step of integrating biological age with other biomarkers. The biological age estimate is combined with other biomarkers of aging, such as telomere length, epigenetic clocks based on promoter CpGs, inflammatory cytokines, and metabolic markers, to generate a comprehensive aging index. The aging index is used to predict the risk of age-related diseases, including cancer, cardiovascular disease, neurodegenerative disease, and diabetes. 

The method further comprises the step of predicting disease risk. The biological age estimate is used to calculate the relative risk of developing specific age-related diseases. For example, individuals with a biological age that exceeds their chronological age by more than 10 years have a 2.5-fold increased risk of developing cancer within the next decade. The method provides a risk score for each major disease category, enabling personalized prevention strategies. 

The method further comprises the step of detecting environmental exposures. The method can detect the impact of UV radiation, smoking, air pollution, and occupational toxins on biological aging. For example, individuals with high UV exposure have a biological age that is 3 to 5 years older than their chronological age, even after adjusting for lifestyle factors. The method can quantify the biological cost of environmental exposures and guide public health interventions. 

The method further comprises the step of detecting the impact of socioeconomic status. Individuals with low socioeconomic status have a biological age that is 4 to 8 years older than their chronological age, reflecting the cumulative biological toll of chronic stress, poor nutrition, and limited healthcare access. The method can quantify health disparities and guide policy interventions. 

The method further comprises the step of detecting the impact of psychological stress. Individuals with chronic stress, anxiety, or depression have a biological age that is 2 to 6 years older than their chronological age, reflecting the biological cost of allostatic load. The method can guide mental health interventions and stress management programs. 

The method further comprises the step of detecting the impact of physical activity. Individuals who engage in regular moderate to vigorous physical activity have a biological age that is 3 to 7 years younger than their chronological age, reflecting the anti-aging effects of exercise. The method can guide fitness programs and public health campaigns. 

The method further comprises the step of detecting the impact of diet. Individuals who follow a Mediterranean diet rich in fruits, vegetables, nuts, and olive oil have a biological age that is 2 to 5 years younger than their chronological age. Individuals who consume a Western diet high in processed foods, sugar, and saturated fat have a biological age that is 3 to 6 years older than their chronological age. The method can guide nutritional counseling and dietary interventions. 

The method further comprises the step of detecting the impact of sleep. Individuals who sleep 7 to 9 hours per night with good sleep quality have a biological age that is 1 to 3 years younger than their chronological age. Individuals with chronic insomnia or sleep apnea have a biological age that is 2 to 5 years older than their chronological age. The method can guide sleep medicine and public health initiatives. 

The method further comprises the step of detecting the impact of mindfulness and meditation. Individuals who practice mindfulness or meditation for at least 20 minutes per day have a biological age that is 1 to 3 years younger than their chronological age. The method can guide integrative medicine and wellness programs. 

The method further comprises the step of detecting the impact of social connection. Individuals with strong social networks and high social support have a biological age that is 2 to 4 years younger than their chronological age. Individuals with social isolation have a biological age that is 3 to 6 years older than their chronological age. The method can guide community health initiatives and elder care programs. 

The method further comprises the step of detecting the impact of education. Individuals with higher levels of education have a biological age that is 1 to 4 years younger than their chronological age, reflecting the protective effects of cognitive reserve. The method can guide lifelong learning and educational policy. 

The method further comprises the step of detecting the impact of alcohol consumption. Individuals who consume moderate amounts of alcohol have a biological age that is 1 to 2 years older than their chronological age. Individuals who consume heavy amounts of alcohol have a biological age that is 5 to 8 years older than their chronological age. The method can guide public health campaigns on alcohol use. 

The method further comprises the step of detecting the impact of tobacco use. Individuals who smoke tobacco have a biological age that is 5 to 10 years older than their chronological age. The method can guide smoking cessation programs and tobacco control policy. 

The method further comprises the step of detecting the impact of obesity. Individuals with a body mass index above 30 have a biological age that is 3 to 7 years older than their chronological age. The method can guide weight management programs and public health initiatives.