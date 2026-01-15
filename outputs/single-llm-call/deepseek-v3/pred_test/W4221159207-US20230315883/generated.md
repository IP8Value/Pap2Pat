# DESCRIPTION

## STATEMENT REGARDING PRIOR DISCLOSURES BY THE INVENTOR OR JOINT INVENTOR

The inventors have not made any prior public disclosures of the invention described herein that would constitute statutory bars under 35 U.S.C. § 102. No publications, public uses, sales, offers for sale, or other disclosures of the invention have been made more than one year prior to the filing date of this application. The invention represents original work by the inventors that has not been previously disclosed in any form that would preclude patentability.

## BACKGROUND

Entity resolution refers to the process of identifying records that correspond to the same real-world entity across multiple datasets. This becomes particularly challenging when the datasets belong to different organizations that cannot directly share sensitive information due to privacy regulations or competitive concerns. Traditional entity resolution techniques require access to the complete datasets, which creates significant privacy risks when dealing with sensitive information such as medical records, financial data, or personal identifiers.

Private set intersection (PSI) is a cryptographic protocol that allows two parties to compute the intersection of their private sets without revealing any additional information about elements not in the intersection. While PSI provides strong privacy guarantees, it only works for exact matches and cannot handle the common scenario where records may represent the same entity but contain variations or errors in the identifying fields.

Locality-sensitive hashing (LSH) is a technique that hashes similar inputs to the same or similar hash values with high probability. Unlike cryptographic hash functions that aim to minimize collisions, LSH functions are designed to maximize collisions for similar inputs while maintaining separation for dissimilar ones. This property makes LSH particularly useful for approximate matching tasks where exact matches cannot be guaranteed.

## SUMMARY

The present invention provides a novel method for performing private data intersection that combines the privacy-preserving properties of private set intersection with the approximate matching capabilities of locality-sensitive hashing. The method begins by having each party perform private set intersection on their respective datasets to identify exact matches. For records that do not exactly match, the system computes locality-sensitive hash values that capture the similarity between records without revealing the underlying data.

The parties then jointly perform a second private set intersection operation on the locality-sensitive hash values to identify records that are sufficiently similar based on their hash collisions. This two-stage approach allows for both exact matching through traditional PSI and approximate matching through LSH-enhanced PSI. The system determines matching records by comparing similarity scores derived from the hash collisions against predefined thresholds.

The invention includes a computer program product comprising a non-transitory computer-readable storage medium with program instructions that, when executed by a processor, cause the processor to implement the private data intersection method. The program instructions include modules for performing private set intersection, computing locality-sensitive hash values, and determining record matches based on similarity thresholds.

A computer system implementing the invention includes one or more processors, memory, and network interfaces configured to execute the private data intersection method. The system facilitates secure communication between parties while maintaining the privacy of non-matching records. The architecture supports both local execution and distributed implementations across networked environments.

## DETAILED DESCRIPTION

Entity resolution and private set intersection (PSI) form the foundation of the present invention's approach to privacy-preserving record linkage. Traditional entity resolution techniques cannot be directly applied when datasets contain sensitive information that cannot be shared between parties. The need for privacy-preserving record linkage has grown significantly with increasing data privacy regulations and cross-organizational data collaboration requirements.

Existing protocols for private record linkage suffer from several limitations. Many approaches require a trusted third party to perform the matching, which introduces additional security risks and operational complexity. Other solutions either reveal too much information about non-matching records or have computational complexity that makes them impractical for large datasets. The present invention addresses these limitations through a novel combination of cryptographic protocols and similarity-preserving hashing techniques.

The Locality-Sensitive Hashing (LSH) algorithm plays a central role in the invention's approach to approximate matching. LSH functions possess several key properties that make them suitable for privacy-preserving applications. First, they provide probabilistic guarantees about collision rates for similar inputs. Second, they allow tuning of the similarity thresholds through parameter selection. Third, they operate independently of the data distribution, unlike some alternative approaches like Bloom filters.

The invention proposes using LSH specifically for private data matching by combining it with private set intersection protocols. This combination provides stronger privacy guarantees than using either technique alone. The LSH computation transforms raw record data into hash values that preserve similarity relationships while obscuring the original content. The private set intersection then allows parties to discover matching hash values without revealing non-matching ones.

Band signatures represent a critical component of the LSH implementation in the invention. Each record is processed through multiple independent hash functions, and the results are grouped into bands. The concatenated hash values within each band form a signature that serves as the basis for similarity comparisons. Two records are considered potential matches if they share at least one band signature, with the probability of such matches being tunable through the selection of band size and count.

The determination of matching records using LSH involves comparing the band signatures generated from each party's dataset. The system counts the number of matching band signatures between record pairs and uses this count to estimate their similarity. Records exceeding a predefined similarity threshold are identified as matches, while those below the threshold are excluded from the results.

Optimization methods improve the accuracy of the matching process by adjusting LSH parameters based on dataset characteristics. The system can compute the Jaccard similarity index between record pairs as a quantitative measure of their similarity. This index represents the ratio of shared features to total features between two records, providing a normalized similarity score between 0 and 1.

Shingles, also known as k-grams, play an important role in computing the Jaccard index. The system breaks each record field into all possible contiguous substrings of length k, creating a set of shingles that capture the field's content at multiple granularities. The size of the intersection and union of these shingle sets between two records directly determines their Jaccard similarity.

The invention proposes using private set intersection to privately compute the Jaccard index between records. This allows parties to determine record similarity without revealing the actual shingles or their counts. The system implements this through secure multi-party computation protocols that preserve the privacy of non-matching records while accurately identifying matches.

System, method, and computer program product embodiments of the invention provide comprehensive solutions for private data intersection. The system embodiment includes specialized hardware and software components optimized for performing privacy-preserving record linkage at scale. The method embodiment defines the step-by-step process for executing the private data intersection, including preprocessing, hashing, and matching phases.

The computer program product embodiment includes a computer-readable storage medium containing executable instructions for implementing the private data intersection method. The storage medium may comprise various types of persistent memory devices, including solid-state drives, magnetic disks, or optical media. The program instructions are structured to optimize performance while maintaining strict privacy guarantees.

Network connectivity enables the downloading of computer-readable program instructions to local systems for execution. The instructions may be transmitted over wired or wireless networks using standard protocols. Upon receipt, the instructions are stored in local memory and executed by the system's processors to perform the private data intersection operations.

The computer-readable program instructions encompass various levels of abstraction, from high-level language statements to low-level machine code. Assembler instructions, instruction set architecture (ISA) commands, and direct machine instructions all represent valid forms of the program instructions. The system stores these instructions in memory hierarchies ranging from processor registers to secondary storage devices.

Electronic circuitry, including application-specific integrated circuits (ASICs) and field-programmable gate arrays (FPGAs), may execute portions of the computer-readable program instructions for performance optimization. These hardware accelerators implement critical cryptographic operations and hash computations with greater efficiency than general-purpose processors.

Flowchart illustrations and block diagrams help visualize the invention's operation and system architecture. Functional block diagrams depict the major components of the network computing environment where the invention operates. These diagrams show the relationships between user devices, servers, and network infrastructure that collectively enable private data intersection.

User devices represent one endpoint in the private data intersection system. These devices include processing units, memory, and interfaces for interacting with users and servers. The user interface components allow data owners to configure matching parameters and review results while maintaining control over their sensitive information.

Graphical user interfaces and application programming interfaces (APIs) provide flexible access to the private data intersection functionality. Users may interact with the system through visual controls or programmatic calls, depending on their technical expertise and application requirements. Scripting interfaces enable automation of common matching tasks across multiple datasets.

The application software securely finds matching records by implementing the invention's core algorithms. It manages the entire workflow from data preprocessing through final match reporting while enforcing strict privacy controls. The application coordinates with server components to perform computationally intensive operations that exceed local device capabilities.

Server components form the backbone of the private data intersection system. These include specialized programs for performing the private set intersection and locality-sensitive hashing operations. The servers provide computational resources to client devices while maintaining the confidentiality of all processed data.

The private data intersection program represents the central server-side component implementing the invention's novel algorithms. It orchestrates the secure matching process across multiple parties while preventing information leakage. The program manages cryptographic key exchanges, protocol executions, and result computations according to predefined privacy parameters.

Servers provide various resources to computing devices participating in the private data intersection. These resources include processing power for cryptographic operations, storage for intermediate results, and bandwidth for secure communications. The system dynamically allocates resources based on workload demands and participant requirements.

Cloud computing environments offer scalable platforms for deploying the private data intersection system. The invention leverages cloud infrastructure to handle variable workloads and large datasets efficiently. Cloud deployment models support both public access and private installations for organizations with specific security requirements.

Clustered computers provide high-performance implementations of the private data intersection algorithms. These clusters distribute computational loads across multiple nodes to achieve near-linear scaling with dataset sizes. Specialized networking between cluster nodes minimizes communication overhead during protocol execution.

The network computing environment connects user devices and servers through secure communication channels. These channels employ strong encryption to protect all transmitted data, including intermediate protocol messages. The network infrastructure supports both local-area and wide-area deployments with appropriate quality-of-service guarantees.

Network interconnections between devices enable the collaborative nature of the private data intersection. Parties maintain their own secure systems while participating in joint computations through the network. The invention's protocols ensure that network communications reveal no sensitive information beyond the intended matching results.

Modifications and variations to the embodiments allow adaptation to different use cases and technical constraints. The system architecture supports customization of cryptographic algorithms, hash functions, and matching thresholds based on application requirements. These variations maintain the core privacy guarantees while optimizing for specific performance or accuracy needs.

The description of the present invention concludes with an overview of the system architecture and its components. The architecture balances privacy, performance, and usability requirements through careful design of its constituent elements. Server components implement the computationally intensive operations while client components focus on user interaction and data management.

Private data intersection programs form the algorithmic core of the server implementation. These programs execute the novel combination of private set intersection and locality-sensitive hashing that distinguishes the invention. They manage the complete workflow from initial setup through final result delivery while enforcing strict privacy controls.

Data processing pipelines transform raw records into formats suitable for private matching. The system applies consistent preprocessing to all datasets to ensure comparability of results. These transformations preserve the semantic content of records while preparing them for cryptographic operations and hashing.

Informed consent mechanisms ensure that data owners understand and authorize the use of their information in private matching operations. The system records consent decisions and enforces them throughout the data processing lifecycle. This includes controls over which fields participate in matching and how results may be used.

Data storage subsystems securely maintain records before, during, and after matching operations. The implementation uses encryption and access controls to protect stored information from unauthorized access. Storage architectures range from local databases to distributed file systems depending on deployment scale and performance requirements.

Storage device implementations vary according to performance and capacity needs. High-speed solid-state storage accelerates cryptographic operations while high-capacity magnetic storage accommodates large datasets. The system optimizes data placement across storage tiers based on access patterns and performance requirements.

Data record databases organize information for efficient processing during private matching. These databases support flexible schemas to accommodate diverse record formats across different organizations. Indexing structures optimize access to specific records during the matching process.

Data preprocessing prepares raw records for private matching through several transformation steps. Cleaning operations correct formatting inconsistencies and standardize representations across datasets. Integration combines information from multiple sources into unified record structures.

Data cleaning addresses quality issues that could affect matching accuracy. The process identifies and corrects typographical errors, inconsistent formatting, and missing values. Automated validation rules ensure records meet minimum quality standards before participating in matching.

Data integration combines information from disparate sources into coherent structures. The system resolves schema differences and semantic conflicts to create unified representations. This enables meaningful comparison of records originating from different organizations or systems.

Data reduction techniques minimize the computational overhead of private matching. The system selects the most discriminative fields for comparison while excluding redundant or irrelevant information. Dimensionality reduction methods preserve matching accuracy while decreasing processing requirements.

Data transformation converts records into formats optimized for private matching operations. These transformations include normalization of text, encoding of categorical values, and scaling of numerical attributes. Consistent transformations across datasets ensure fair comparison of record similarities.

Joining fields combines related attributes to improve matching accuracy. The system intelligently concatenates fields that together provide stronger identity signals than individual attributes. This includes combining name components or address elements into unified comparison units.

Canonicalization processes standardize record representations to minimize spurious differences. The system applies consistent formatting rules for dates, names, and other commonly varied fields. This reduces false mismatches caused by formatting variations rather than substantive differences.

Cyclic shingles extend the basic shingling approach to capture rotational variations in text. By considering circular permutations of character sequences, the system becomes more robust to offset differences in field values. This improves matching accuracy for fields where content may shift positionally.

Jaccard similarity provides the mathematical foundation for comparing record similarity in the invention. The coefficient measures the overlap between sets of features extracted from records. Its normalization between 0 and 1 enables consistent thresholding across different record types and sizes.

Private set intersection enables the secure discovery of matching features between records. The cryptographic protocol reveals only the existence of matches while concealing non-matching elements. This forms the basis for computing Jaccard similarity without exposing the underlying record contents.

PSI on record fields applies private set intersection to specific attributes within records. The system can perform separate PSI operations on different fields to enable flexible matching strategies. Field-specific PSI allows tuning of matching precision across different attribute types.

Locality-sensitive hash computation transforms record features into comparable representations. The system applies multiple independent hash functions to each feature set, creating a comprehensive similarity profile. These hash values enable efficient comparison while preserving privacy.

LSH computation parameters control the trade-off between matching accuracy and privacy. The number of hash functions and their configuration determine the probability of collision for similar records. Careful parameter selection optimizes the balance between false matches and missed matches.

Min-hash values represent the core output of the locality-sensitive hashing process. Each min-hash corresponds to the minimal hash value across a record's feature set for a particular hash function. The collection of min-hash values forms a compact similarity fingerprint for the record.

Band signature creation groups min-hash values into comparable units. The system divides the min-hash vector into contiguous segments called bands, each containing multiple min-hash values. These bands provide multiple independent opportunities for records to demonstrate their similarity.

LSH tuple creation packages band signatures into structured representations for comparison. Each tuple contains the concatenated min-hash values for a band along with metadata about its position. These tuples serve as the atomic units for private set intersection operations.

Match declaration occurs when two records share at least one identical band signature. The system compares LSH tuples between datasets and identifies collisions indicating potential matches. Statistical analysis determines whether these collisions likely represent genuine record similarities.

Jaccard similarity coefficient computation quantifies the strength of potential matches. The system estimates the Jaccard index based on the number of matching band signatures between records. This estimation provides a more nuanced similarity measure than binary match declarations.

Match control mechanisms regulate the flow of information during private matching. The system enforces strict policies about what matching information becomes visible to which parties. These controls prevent unintended disclosure of sensitive information through the matching process.

Jaccard threshold selection determines the minimum similarity required for match acceptance. Users can adjust this threshold based on their tolerance for false matches versus missed matches. Higher thresholds increase precision but may decrease recall in the matching results.

Probability of match calculations estimate the likelihood that declared matches represent true similarities. The system computes these probabilities based on the number of matching bands and their statistical properties. This helps users assess the reliability of matching results.

Optimization problem formulation frames the parameter selection as a mathematical optimization. The system seeks to maximize matching accuracy while minimizing privacy loss and computational cost. Constraint programming techniques help navigate the complex trade-offs between these objectives.

Empirical search methods explore the parameter space to identify optimal configurations. The system can test different LSH parameter combinations on sample data to measure their actual performance. This data-driven approach complements theoretical analysis of parameter effects.

Analytical formulas for b and r provide theoretical guidance for parameter selection. These formulas relate the number of bands (b) and rows per band (r) to desired similarity thresholds and collision probabilities. The system uses these relationships to initialize parameter values before empirical refinement.

Private data intersection program implementation brings together all algorithmic components into executable software. The program coordinates the sequence of operations from data input through result output while maintaining strict privacy controls. Modular design allows customization for different application scenarios.

Purpose of private data intersection defines the core objectives of the invention's method. The primary goal is to identify matching records across private datasets without revealing non-matching information. Secondary goals include computational efficiency, matching accuracy, and operational flexibility.

Motivation for private data intersection stems from growing needs for collaborative data analysis under privacy constraints. Organizations increasingly require methods to combine insights from their respective datasets without compromising sensitive information. The invention addresses this need through cryptographically secure protocols.

Embodiment of private data intersection describes the practical instantiation of the invention's concepts. The embodiment includes all necessary components to perform private matching in real-world settings. This covers data handling, protocol execution, and result delivery mechanisms.

PSI protocol implementation details the specific cryptographic techniques used for private set intersection. The invention employs protocols based on commutative encryption that allow secure computation of set intersections. These protocols provide strong privacy guarantees with reasonable computational overhead.

Reordering process in PSI protocol enhances privacy by randomizing the sequence of compared elements. The system applies permutations to the inputs of PSI operations to prevent inference from positional information. This protects against certain types of statistical attacks on the matching process.

Combining cardinality computation with scoring scheme enriches the matching results with quantitative measures. The system not only identifies matching records but also provides similarity scores indicating match strength. This additional information helps users interpret and utilize the matching results effectively.

Private data intersection program for multiple parties extends the basic two-party protocol to collaborative scenarios. The system coordinates simultaneous participation of several organizations in joint matching operations. Multi-party protocols ensure that no single party gains disproportionate information about others' data.

Generating hashes from records constitutes the first step in the private matching process. The system applies cryptographic hash functions to record fields to create secure representations. These hashes preserve the necessary information for matching while concealing the original content.

Applying secret keys to hashes implements the commutative encryption scheme. Each party encrypts their hash values with a private key before sharing them. The encryption allows secure comparison while preventing reconstruction of the original hashes by other parties.

Distributing signed collections enables verification of data authenticity during matching. The system attaches digital signatures to hash collections to prove their origin and integrity. This prevents tampering with the data during transmission and processing.

Searching for band signatures implements the core matching operation. The system compares encrypted band signatures between datasets to identify potential matches. This search occurs in the encrypted domain to maintain privacy throughout the process.

Configuring rules for record matching allows customization of the matching criteria. Users can specify which fields participate in matching and their relative importance. The system translates these rules into parameter settings for the LSH and PSI operations.

Applying rules for record matching implements the user-specified criteria during processing. The system weights different fields according to their matching importance and combines their contributions appropriately. This ensures the matching results align with user expectations and requirements.

Applying screen candidate with exact value provides an initial filtering step. The system first checks for exact matches on high-confidence fields before proceeding to approximate matching. This optimization reduces the computational load of the more expensive LSH operations.

Applying hash algorithms and PSI combines the two core techniques of the invention. The system sequences hash computation and private set intersection to achieve both privacy and approximate matching. Careful coordination ensures these operations complement rather than interfere with each other.

Removing candidates from database optimizes the matching workflow. The system eliminates clearly non-matching records early in the process to focus resources on plausible matches. This pruning step improves both computational efficiency and result quality.

Using similarity-based PSI extends traditional private set intersection with approximate matching capabilities. The invention's novel protocol allows parties to discover records that are similar according to defined metrics, not just exactly equal. This significantly expands the applicability of private matching techniques.

Flow chart diagram for determining intersecting data visualizes the invention's matching process. The diagram shows the sequence of operations from data preparation through final match determination. Decision points highlight where parameter choices affect the workflow path.

Pre-processing of record sets prepares the data for private matching. This includes cleaning, normalization, and feature extraction steps that standardize the representation across datasets. Consistent pre-processing ensures fair comparison during subsequent matching stages.

Performing private set intersection implements the exact matching phase of the protocol. The system securely compares record identifiers or other high-confidence fields to discover unambiguous matches. These exact matches serve as anchors for the more challenging approximate matches.

Computing locality sensitive hash values transforms record contents into similarity-preserving representations. The system applies multiple LSH functions to each record to create comprehensive similarity profiles. These hash values enable efficient comparison while maintaining privacy.

Jointly performing private set intersection on LSH values implements the approximate matching phase. Parties collaborate to discover records with similar hash values without revealing their actual content. This step identifies potential matches that warrant closer similarity assessment.

Determining matching records based on similarity score finalizes the matching results. The system compares the number of matching LSH bands against predefined thresholds to declare valid matches. Statistical analysis ensures these declarations meet desired confidence levels.

Table data for determining intersecting pairs organizes matching results for analysis and verification. The system presents matching record pairs along with their similarity scores and supporting evidence. This structured output facilitates review and use of the matching results.

Generating min-hash values implements the first stage of LSH computation. For each hash function, the system identifies the minimal hash value across all record features. These minima serve as compact representatives of the record's content for similarity purposes.

Grouping min-hashes into bands organizes the hash values for efficient comparison. The system divides the vector of min-hash values into contiguous segments that provide independent similarity evidence. This banding strategy amplifies the probability of detecting true matches while controlling false positives.

Forming LSH for compared pair of records creates the final similarity representation. Each record's LSH consists of the concatenated min-hash bands that will be compared against other records. The system ensures consistent formatting of these representations to enable accurate comparison.

Using separate hash to hash signatures adds an additional privacy layer. The system applies a cryptographic hash function to the band signatures before comparison. This prevents potential inference attacks that might exploit the structure of raw LSH values.

Considering strings in same bucket as similar implements the core LSH matching principle. The system treats records that hash to the same bucket for any band as potential matches. This probabilistic approach balances matching recall with computational efficiency.

Block diagram of computing device illustrates the hardware platform for implementing the invention. The diagram shows the major components including processors, memory, storage, and interfaces that collectively enable private data intersection. Different device configurations support varying scales of operation.

Components of computing device include the physical elements necessary for protocol execution. Processors perform cryptographic and hashing operations, memory stores intermediate results, and network interfaces facilitate secure communication. Specialized hardware accelerators may enhance performance for critical operations.

Operating computing device involves managing the resources for efficient private matching. The system allocates processor cycles, memory, and bandwidth to balance performance across concurrent matching tasks. Resource monitoring ensures stable operation under varying workloads.

Storing program instructions maintains the executable code for ready access. The system keeps frequently used instructions in fast memory while archiving less critical code in secondary storage. Memory hierarchies optimize the trade-off between access speed and storage capacity.

Using removable media provides flexible deployment options for the invention. The system can distribute program instructions on portable storage devices for installation on isolated systems. This supports secure deployment in environments with restricted network access.

Providing input and output interfaces the system with users and external data sources. Input mechanisms accept configuration parameters and raw datasets, while output channels deliver matching results. The system validates all I/O operations to ensure data integrity and security.

Displaying data to user presents matching results in comprehensible formats. The system transforms raw matching data into visualizations and reports tailored to user needs. Interactive interfaces allow exploration and verification of matching outcomes.

Cloud computing environment offers scalable infrastructure for large-scale private matching. The invention leverages cloud resources to handle variable workloads and dataset sizes efficiently. Cloud deployment provides elasticity in resource allocation and geographic distribution.

Characteristics of cloud computing that benefit the invention include on-demand resource provisioning and pay-per-use pricing. The system dynamically scales its cloud resources to match current processing requirements. This optimizes both performance and cost for private matching operations.

Service models of cloud computing support different deployment scenarios for the invention. Infrastructure-as-a-Service provides raw computing resources, Platform-as-a-Service offers development environments, and Software-as-a-Service delivers turnkey solutions. The invention can operate across all these models depending on user needs.

Deployment models of cloud computing accommodate varying security requirements. Public clouds offer cost-effective solutions for less sensitive data, while private clouds provide enhanced control for regulated environments. Hybrid models combine aspects of both for balanced implementations.

Block diagram of cloud computing environment illustrates the distributed architecture supporting the invention. The diagram shows the interconnection of compute nodes, storage systems, and networking components that collectively enable scalable private matching. Load balancing and fault tolerance mechanisms ensure reliable operation.

Communicating with cloud computing nodes implements the distributed nature of the protocol. The system coordinates operations across multiple cloud instances to parallelize computationally intensive tasks. Secure channels protect all inter-node communications from eavesdropping or tampering.

Functional abstraction model layers organize the cloud implementation into logical tiers. The hardware layer provides physical resources, the virtualization layer abstracts these resources, and the service layer delivers matching functionality. This layered approach simplifies management and maintenance of the system.

Layers and functions of cloud computing environment partition responsibilities for efficient operation. Infrastructure management handles resource provisioning, platform services support application development, and matching software implements the core invention algorithms. Clear separation of concerns enhances system reliability and scalability.