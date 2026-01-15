- **Parameter Translation Success**: After validating the success of parameter translation between species, we applied rank-normalized inter-species classifiers to predict synthetic lethality (SL) pairs in humans and mice. This approach leveraged the robustness of SL predictions from well-characterized model organisms like *S. cerevisiae*.

- **Human SL Predictions**: We filtered human gene pairs by annotating VCF files from studies of patients with homozygous deleterious mutations. Gene pairs where both genes were significantly mutated in at least one but no more than 5% of patients were flagged as confirmed non-synthetic lethal (NSL) pairs.

- **Pathway Enrichment**: We identified putative SL pairs with SINaTRA scores >0.95, 0.90, and 0.80, mapping them to KEGG pathways. Fisher's exact test was used to assess if these pairs were significantly enriched in the same pathway compared to random groups of similar size.

- **Protein Complex Enrichment**: We analyzed protein complexes from the CORUM database, focusing on those with unambiguous mappings to Entrez gene IDs. Randomly selected complexes and their intra-complex gene pairs were compared to inter-complex pairs using the Mann-Whitney U test to determine if SL pairs were significantly enriched within complexes.

- **Context-Specific SL**: Using protein expression data from the Protein Atlas, we identified context-specific SL pairs by filtering out those where both proteins were not detected in a given tissue or cell line. The number of retained SL pairs was compared to expected values using the Mann-Whitney U test.

- **Comparison to Published Methods**: We mapped SL predictions from Syn-Lethality and DAISY papers to Entrez gene IDs, comparing their SINaTRA scores to random pairs using the Mann-Whitney U test. Classifiers were constructed for these methods using SINaTRA scores as features and status in the dataset as classes.

- **Human SL Landscape**: Gene pairs with SINaTRA scores ≥0.95 were mapped to Reactome pathways, visualized in Cytoscape. Networks were created where nodes (genes) were connected if their SINaTRA score was ≥0.95, colored by closeness centrality and sized by node degree. Pathway-specific networks were joined based on inter-pathway SL pairs.

- **Function-Specific Mechanisms**: We analyzed gene pairs from major Reactome pathways, calculating median values of node-pair and single-node parameters. Heat maps of the ratio of SL to all gene parameters were created, clustering functions by node-pair parameters. Enrichment for complex or parallel annotations was tested using Fisher's exact test.

- **Cancer Therapies**: We mapped genes from high-SINaTRA pairs (SL score > 0.85) to cancer drugs in the DCDB, identifying unique genes and their pairwise combinations. Heat maps of SINaTRA scores were clustered, with known single-drug and combination therapies overlaid. Statistical tests confirmed that filtered gene pairs did not overlap with cancer drug targets.

- **Statistical Analysis**: Network parameters were calculated using NetworkX 1.8.1, while statistical analysis was performed in R 3.0.2. De Long’s test for ROC curve comparisons was implemented using the pROC library. Python scripts (version 2.7.5) and Matplotlib were used for data processing and visualization, respectively. BioGrid release 3.2.104 provided the necessary interaction data.