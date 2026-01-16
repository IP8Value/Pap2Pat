# DESCRIPTION

## TECHNICAL DOMAIN

The present invention pertains to the field of biotechnology and specifically relates to the genetic engineering of the filamentous fungus *Trichoderma reesei* for the enhanced production of cellulases. The invention further encompasses a method for producing cellulases using a genetically engineered *T. reesei* strain and a low-cost, molasses-based medium. The cellulases produced according to the invention are particularly useful in the enzymatic deconstruction of lignocellulosic biomasses for the production of biofuels and value-added chemicals.

## PRIOR ART

The industrial production of enzymes, particularly cellulases, is a critical component of the bioeconomy, driving the transition from fossil fuels to renewable resources. Cellulases are essential for the conversion of lignocellulosic materials into fermentable sugars, which can then be used to produce biofuels and other valuable chemicals. The ascomycete fungus *Trichoderma reesei* is one of the most widely used platforms for the production of cellulases due to its natural ability to secrete a diverse array of carbohydrate-active enzymes (CAZymes) that efficiently break down lignocellulose.

Historically, the development of hypercellulolytic strains of *T. reesei* has relied on random mutagenesis, leading to the creation of strains like RUT-C30, which exhibits a partial relief of carbon catabolite repression (CCR) by glucose and can achieve extracellular protein titers of up to 20 g/L. However, the enzyme production capabilities of these strains fall short of the industrially relevant levels observed in proprietary strains, which can reach titers exceeding 100 g/L. The genetic modifications responsible for these high titers are often proprietary and not publicly disclosed, creating a significant barrier to the development of new, competitive strains.

Several genetic modifications have been reported to enhance cellulase production in *T. reesei*. These include the constitutive expression of the transcription factor XYR1, which regulates cellulase gene expression, and the introduction of point mutations in XYR1 to further relieve CCR. Other strategies involve the deletion of protease-encoding genes to reduce proteolytic degradation of secreted enzymes and the expression of heterologous enzymes to complement the native enzyme cocktail. Despite these advancements, the combined effects of multiple genetic modifications have been less frequently explored due to technical challenges in genetic engineering, such as low transformation efficiencies and a limited number of selectable markers.

Recent advances in CRISPR/Cas9 technology have significantly simplified the genetic engineering of *T. reesei*, enabling the precise and efficient introduction of multiple modifications in a single strain. This has opened new possibilities for the rational design of hyperproducing *T. reesei* strains. However, the development of such strains must also consider the cost-effectiveness of the production process. Traditional media components, such as yeast extract, are expensive and can significantly increase the overall cost of enzyme production. Therefore, there is a need for a low-cost, industrially viable medium that can support high levels of cellulase production.

## SUMMARY OF THE INVENTION

The present invention addresses the need for a *Trichoderma reesei* strain capable of producing high titers of cellulases and a cost-effective method for their production. Specifically, the invention provides a genetically engineered *T. reesei* strain, designated as Br_TrR03, which incorporates a combination of genetic modifications designed to enhance cellulase production and enable the use of a low-cost, molasses-based medium.

The genetic modifications introduced in the Br_TrR03 strain include:
1. The insertion of the β-glucosidase-encoding gene *cel3a* from *Talaromyces emersonii* under the control of the *xyn1* promoter, replacing the *slp1* gene.
2. The replacement of the *ace1* gene with a mutated *xyr1* allele (V821F) driven by the *pdc1* promoter.
3. The insertion of the invertase-encoding gene *suc1* from *Aspergillus niger* driven by its native regulatory sequences, replacing the *pep1* gene.

These modifications collectively aim to:
- Increase the β-glucosidase activity of the secreted enzyme cocktail.
- Enhance the transcription levels of genes encoding secreted enzymes.
- Reduce the proteolytic activity of the secretome.
- Enable the utilization of sucrose as a carbon source.
- Abolish the need for inducing sugars for enzyme production.

The invention further provides a method for producing cellulases using the Br_TrR03 strain and a molasses-based medium. The method involves cultivating the strain in a bioreactor with a batch medium containing molasses, ammonium sulfate, and molasses-grown yeast cells, followed by a fed-batch phase with molasses. This process enables the production of cellulases at a titer of 80.6 g/L (0.24 g/L/h), which is the highest experimentally supported titer reported for *T. reesei*.

The cellulase cocktail produced by the Br_TrR03 strain exhibits high specific enzymatic activities and superior saccharification efficiencies compared to the parental RUT-C30 strain and a commercially available cellulase preparation. The use of a molasses-based medium and molasses-grown yeast cells as a nitrogen source significantly reduces the production costs, making the process economically viable for industrial applications.

## DESCRIPTION OF THE EMBODIMENTS

### Examples

#### Example 1: Rational Engineering of *T. reesei* RUT-C30

To create the Br_TrR03 strain, a single-plasmid CRISPR/Cas9 system and markerless donor cassettes were employed. The CRISPR/Cas9 plasmid, pTrCas9gRNA1, was constructed to carry a codon-optimized *Streptococcus pyogenes* Cas9 gene, a ribozyme-mediated guide RNA (gRNA) cassette, an antibiotic resistance selection marker, and the AMA1 fungal replicator sequence. This system was used to introduce the following genetic modifications in the RUT-C30 strain:

1. **Insertion of *cel3a* Gene**: The β-glucosidase-encoding gene *cel3a* from *Talaromyces emersonii* was placed under the control of the *xyn1* promoter and integrated into the *slp1* locus. This modification significantly increased the β-glucosidase activity of the secreted enzyme cocktail.

2. **Replacement of *ace1* Gene**: The *ace1* gene, a repressor of cellulase gene expression, was replaced with a mutated *xyr1* allele (V821F) driven by the *pdc1* promoter. This modification enhanced the transcription levels of cellulase-encoding genes and relieved CCR by glucose.

3. **Insertion of *suc1* Gene**: The invertase-encoding gene *suc1* from *Aspergillus niger* was inserted into the *pep1* locus, enabling the utilization of sucrose as a carbon source.

The resulting strain, Br_TrR03, was verified through PCR and Sanger sequencing to confirm the successful integration of the genetic modifications.

#### Example 2: Development of a Molasses-Based Cellulase Production Process

To develop a cost-effective process for cellulase production, the Br_TrR03 strain was cultivated in a bioreactor with a molasses-based medium. The batch medium composition was as follows:
- 20.0 g/L (NH4)2SO4
- 30.0 g/L sugars from sugarcane molasses
- 20.0 g/L yeast-derived organic nitrogen source (yeast extract, YEPD-grown yeast, or molasses-grown yeast)

The fed-batch phase involved the continuous feeding of sugarcane molasses at a rate of 1.0 g/L of total sugars per hour. The process parameters, including aeration, dissolved oxygen (DO), pH, and temperature, were carefully controlled to optimize enzyme production.

The Br_TrR03 strain was able to produce cellulases at a titer of 80.6 g/L (0.24 g/L/h) in 336 hours. The enzyme cocktail exhibited high specific enzymatic activities and superior saccharification efficiencies, particularly in the deconstruction of industrially pretreated sugarcane straw.

#### Example 3: Characterization of Enzyme Cocktails

The cellulase cocktails produced by the Br_TrR03 strain and the parental RUT-C30 strain were characterized for their enzymatic activities and saccharification efficiencies. The specific activities of cellobiohydrolase, β-glucosidase, endoglucanase, xylanase, and β-xylosidase were measured using standard assays. The Br_TrR03 cocktail showed significant improvements in specific enzymatic activities, particularly in β-glucosidase and xylanase activities.

Saccharification assays were conducted using the pretreated sugarcane straw at a high solids content (20% w/w). The Br_TrR03 cocktail released more glucose, xylose, and overall sugars compared to the RUT-C30 cocktail, demonstrating its superior saccharification efficiency. The addition of the whole fermentation broth from the Br_TrR03-MMGY fermentation did not affect the saccharification efficiency, suggesting that the whole broth can be used directly in industrial applications.

#### Example 4: Techno-Economic Analysis

A detailed techno-economic analysis was conducted to assess the cost-competitiveness of the cellulase cocktail produced by the Br_TrR03 strain. The analysis considered the cost of raw materials, energy consumption, labor, and capital investment. The use of a molasses-based medium and molasses-grown yeast cells as a nitrogen source significantly reduced the production costs, making the process economically viable for industrial-scale production.

In conclusion, the Br_TrR03 strain and the molasses-based production process represent a significant advancement in the field of industrial enzyme production. The high titers of cellulases, superior enzymatic activities, and cost-effective production process make the Br_TrR03 strain an attractive platform for the enzymatic deconstruction of lignocellulosic biomasses in biofuel and chemical production.