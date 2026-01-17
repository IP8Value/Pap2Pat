# DESCRIPTION

## GOVERNMENT SPONSORSHIP

This invention was made with government support under Grant No. [Insert Grant Number] awarded by [Insert Funding Agency]. The government has certain rights in the invention.

## TECHNICAL FIELD

The present invention relates to methods and systems for achieving super-Mendelian inheritance of genetic traits in organisms. More specifically, the invention provides novel gene-drive systems that enable the efficient and precise dissemination of beneficial genetic variants, such as point mutations, across populations. The invention is particularly useful in the fields of genetics, biotechnology, and agriculture, where the controlled propagation of desirable traits is of significant importance.

## BACKGROUND

Efficient super-Mendelian inheritance of transgenic insertional elements has been demonstrated in various model organisms, including flies, mosquitoes, yeast, and mice. While numerous potentially impactful applications of gene-drive systems have been proposed, they are currently limited to copying relatively large DNA cargo sequences (approximately 1–10 kilobases). Many desired genetic traits, such as drought tolerance in plants, increased crop yield, pest resistance, or insecticide sensitivity, result from allelic variants that alter only one or a few base pairs. An efficient system for super-Mendelian inheritance of such subtle genetic variants would significantly accelerate efforts to disseminate favorable traits throughout populations or to assemble complex genotypes consisting of point-mutant alleles in combination with insertional transgenes for a multitude of research and applied purposes.

Current CRISPR-Cas9-based gene editing approaches involve enzymatic cleavage of a sensitive allele and repair by copying information from an exogenously provided cut-resistant oligonucleotide or double-stranded DNA template. However, these methods are limited in their ability to achieve super-Mendelian inheritance of subtle genetic variants. There is a need for a more versatile and efficient system that can drive the inheritance of beneficial allelic variants without the limitations of existing gene-drive technologies.

## SUMMARY OF THE INVENTION

The present invention addresses the aforementioned limitations by providing a novel gene-drive system that enables the efficient super-Mendelian inheritance of beneficial allelic variants. The invention comprises two forms of allelic-drive: copy-cutting and copy-grafting.

1. **Copy-Cutting**: This form of allelic-drive involves a Cas9-gRNA complex selectively cutting one allelic variant, followed by homology-directed repair (HDR)-mediated repair and replacement with a non-cleavable allele of the same gene provided in trans. The gRNA is designed to target a specific site on the non-preferred allele, while the preferred allele is resistant to cleavage due to a mutation in the protospacer adjacent motif (PAM) site or within the gRNA target sequence.

2. **Copy-Grafting**: This form of allelic-drive involves copying a short genomic interval that encompasses a favored allele in proximity to a gRNA cut site. The favored allele is associated with neighboring sequences resistant to gRNA cleavage. This method is more broadly applicable as it does not require the preferred allele to lack a PAM site or differ in the core gRNA sequence.

The invention further includes a dual gRNA drive system that combines a gene-drive element with a second gRNA to direct selective cleavage of a non-preferred allele at a separate genomic site. This dual gRNA drive system results in efficient super-Mendelian inheritance of both the gene-drive element and the beneficial allelic variant via germline transmission.

Key features of the invention include:
- **High Efficiency**: The allelic-drive systems achieve high rates of super-Mendelian inheritance, significantly exceeding the efficiency of traditional gene-drive elements.
- **Versatility**: The copy-grafting method is broadly applicable to various gene targets and can be used in diverse experimental and agricultural contexts.
- **Lethal Mosaicism**: The invention leverages the phenomenon of lethal mosaicism to eliminate non-functional NHEJ-induced alleles, ensuring the persistence of the preferred allele.
- **Shadow-Drive**: Perdurable Cas9-gRNA complexes transmitted maternally for one generation in the absence of the Cas9 or gRNA transgenes extend gene- or allelic-drive for an additional generation.

The invention has numerous potential applications, including the aggregation of multiple favored naturally occurring allelic variants in plants and animals, the reversal of pesticide resistance in pest species, and the prevention of host species from serving as disease vectors or pests.

## DETAILED DESCRIPTION

### Overview

The present invention provides novel gene-drive systems that enable the efficient super-Mendelian inheritance of beneficial allelic variants. The invention comprises two forms of allelic-drive: copy-cutting and copy-grafting. These systems are designed to overcome the limitations of existing gene-drive technologies and provide a more versatile and efficient means of disseminating favorable traits across populations.

### Copy-Cutting

#### Mechanism

Copy-cutting involves a Cas9-gRNA complex selectively cutting one allelic variant, followed by homology-directed repair (HDR)-mediated repair and replacement with a non-cleavable allele of the same gene provided in trans. The gRNA is designed to target a specific site on the non-preferred allele, while the preferred allele is resistant to cleavage due to a mutation in the protospacer adjacent motif (PAM) site or within the gRNA target sequence.

#### Example

In the context of the Drosophila Notch (N) locus, a gRNA (gRNA-N+) was designed to target the wild-type N+ allele but not the NAx16 allele, which lacks a PAM site due to a Gly → Arg amino acid substitution. The gRNA-N+ was incorporated into a DsRed-marked gRNA-only "CopyCat" element (ccN) designed to insert into and copy itself at the closely linked yellow locus. In the presence of an unlinked source of Cas9, the DsRed-marked y<ccN> element copied itself at the yellow locus and resulted in super-Mendelian inheritance of the gRNA-insensitive NAx16 allele via copy-cutting.

### Copy-Grafting

#### Mechanism

Copy-grafting involves copying a short genomic interval that encompasses a favored allele in proximity to a gRNA cut site. The favored allele is associated with neighboring sequences resistant to gRNA cleavage. This method is more broadly applicable as it does not require the preferred allele to lack a PAM site or differ in the core gRNA sequence.

#### Example

In the context of the Drosophila Notch (N) locus, a wild-type N+IS allele was recombined with the y<ccN> CopyCat element. The N+IS allele carries a single-nucleotide change (C → A) at the -4 position, making it resistant to cleavage by the gRNA-N+. The y<ccN> CopyCat element was recombined with the wild-type N+IS allele and placed in trans to a sensitive NAxE2 allele. The inverse allelic-drive via copy-grafting resulted in the efficient conversion of the NAxE2 allele to the N+IS allele.

### Dual gRNA Drive System

The invention further includes a dual gRNA drive system that combines a gene-drive element with a second gRNA to direct selective cleavage of a non-preferred allele at a separate genomic site. This dual gRNA drive system results in efficient super-Mendelian inheritance of both the gene-drive element and the beneficial allelic variant via germline transmission.

### Key Features

#### High Efficiency

The allelic-drive systems achieve high rates of super-Mendelian inheritance, significantly exceeding the efficiency of traditional gene-drive elements. For example, in Drosophila, the copy-cutting method achieved a conversion rate of 79.5%, while the copy-grafting method achieved a conversion rate of 78.2%.

#### Versatility

The copy-grafting method is broadly applicable to various gene targets and can be used in diverse experimental and agricultural contexts. This method can be applied to plants and animals to combine favorable traits, such as drought tolerance, higher yields, and pest resistance.

#### Lethal Mosaicism

The invention leverages the phenomenon of lethal mosaicism to eliminate non-functional NHEJ-induced alleles, ensuring the persistence of the preferred allele. Lethal mosaicism occurs when maternally perduring Cas9-gRNA complexes cause a large enough proportion of somatic cells to have two mutant copies of the gene, leading to embryonic lethality.

#### Shadow-Drive

Perdurable Cas9-gRNA complexes transmitted maternally for one generation in the absence of the Cas9 or gRNA transgenes extend gene- or allelic-drive for an additional generation. This shadow-drive phenomenon acts as a genetic slingshot to enhance the spread of the preferred allele.

### Applications

#### Aggregation of Multiple Favored Alleles

The allelic-drive systems can be used to aggregate multiple favored naturally occurring allelic variants in plants and animals. For example, in polyploid crops, the systems can be used to combine several preferred alleles providing drought resistance, higher yields, optimal architectures, or more rapid growth.

#### Reversal of Pesticide Resistance

The allelic-drive systems can be used to reverse pesticide resistance in pest species. By targeting essential components of the nervous system, such as the Na+ channel or glutamate receptor, the systems can revert populations back to their wild-type-sensitive state.

#### Prevention of Disease Vectors and Pests

The allelic-drive systems can be used to favor genetic variants that prevent host species from serving as disease vectors or pests. This can have significant impacts on disease-reduction strategies and agricultural practices.

### Examples

#### Example 1: Copy-Cutting in Drosophila

**Materials and Methods**

- **Construction of ccN CopyCat Element**: The ccN CopyCat plasmid was constructed using homology arms to the yellow locus abutting the gRNA-y1 cleavage site and carrying gRNA-y1, gRNA-N+, and a 3XP3-DsRed eye marker. The plasmid was transformed into One Shot® TOP10 competent cells and purified using the Qiagen Plasmid Midi Kit. The ccN plasmid was injected into embryos collected from a wa NAx16 rb− stock with a transient source of pHsp70-Cas9. Male transformants carrying the ccN element were identified by their yellow− and DsRed fluorescent eye-marker phenotypes.

- **Crossing Scheme**: F0 females bearing the y<ccN> and NAx16 alleles were crossed to wild-type males homozygous for a Cas9 source on the third chromosome to generate F1 master females. F1 master females were crossed to wild-type males carrying a normal X chromosome or the multiply inverted FM7 balancer chromosome. The resulting F2 progeny were scored for transmission of the DsRed-marked y<ccN> element and the NAx16 allele.

**Results**

- **Transmission Percentages**: Transmission percentages for the y<ccN> (DsRed+) and dominant NAx16 alleles in F2 males revealed highly biased inheritance of both alleles, with 85.3% of progeny being DsRed+ and 93.6% being NAx16. Sequence analysis of individuals from non-DsRed lines revealed that they all carried NHEJ-induced loss-of-function mutations.

- **Phenotypic Analysis**: Phenotypic analysis of F2 females revealed a high degree of mosaicism, with wings often displaying a mixture of wild-type, gain-, and loss-of-function phenotypes. Individual F2 female lines were established to permit unambiguous scoring of N phenotypes in subsequent generations.

#### Example 2: Copy-Grafting in Drosophila

**Materials and Methods**

- **Construction of N+IS Allele**: A cleavage-insensitive N+IS allele was recombined with the y<ccN> CopyCat element. The N+IS allele carries a single-nucleotide change (C → A) at the -4 position, making it resistant to cleavage by the gRNA-N+.

- **Crossing Scheme**: F1 master females carrying the wild-type N+IS allele in trans to the sensitive NAxE2 allele were crossed to N+IS males. F2 progeny were scored based on inheritance of the y<ccN> wa N+IS donor chromosome or the w+ receiver chromosome based on their eye color phenotype.

**Results**

- **Conversion Rates**: The efficiency of inverse allelic-drive via copy-grafting was comparable to that observed by copy-cutting, with 78.2% conversion of receiver chromosomes in males. The system also resulted in the generation of N− alleles and a dearth of receiver chromosomes in F2 males.

### Conclusion

The present invention provides novel gene-drive systems that enable the efficient super-Mendelian inheritance of beneficial allelic variants. The invention comprises two forms of allelic-drive: copy-cutting and copy-grafting. These systems are designed to overcome the limitations of existing gene-drive technologies and provide a more versatile and efficient means of disseminating favorable traits across populations. The invention has numerous potential applications in genetics, biotechnology, and agriculture, and represents a significant advancement in the field of active genetics.

### References

1. [Insert Reference 1]
2. [Insert Reference 2]
3. [Insert Reference 3]
4. [Insert Reference 4]
5. [Insert Reference 5]
6. [Insert Reference 6]
7. [Insert Reference 7]
8. [Insert Reference 8]
9. [Insert Reference 9]
10. [Insert Reference 10]
11. [Insert Reference 11]
12. [Insert Reference 12]
13. [Insert Reference 13]
14. [Insert Reference 14]
15. [Insert Reference 15]
16. [Insert Reference 16]
17. [Insert Reference 17]
18. [Insert Reference 18]
19. [Insert Reference 19]
20. [Insert Reference 20]
21. [Insert Reference 21]
22. [Insert Reference 22]
23. [Insert Reference 23]
24. [Insert Reference 24]
25. [Insert Reference 25]
26. [Insert Reference 26]
27. [Insert Reference 27]
28. [Insert Reference 28]
29. [Insert Reference 29]
30. [Insert Reference 30]
31. [Insert Reference 31]
32. [Insert Reference 32]
33. [Insert Reference 33]
34. [Insert Reference 34]
35. [Insert Reference 35]

### Reporting Summary

Further information on experimental design is available in the Nature Research Reporting Summary linked to this article.