## FIELD

- relate to anonymizing datasets

The present invention relates to systems and methods for anonymizing structured datasets, particularly binary matrices representing user-feature interactions, while preserving utility for downstream analytical tasks. The invention provides a computationally efficient and provably approximate algorithmic framework for transforming raw datasets into privacy-preserving forms that satisfy a refined notion of anonymity known as smooth-k-anonymity. This framework is specifically designed for sparse, high-dimensional binary data commonly found in digital user behavior logs, social networks, recommendation systems, and advertising platforms. Unlike prior approaches that rely on suppression or noise injection, the invention enables the controlled addition and removal of feature associations to achieve anonymity with minimal distortion to the underlying data structure. The method operates on bipartite graphs where one partition represents users and the other represents features, with edges indicating presence or absence of associations. The invention is implemented as a software-based computing system that processes large-scale datasets in polynomial time, offering practical scalability for real-world applications involving billions of user records and hundreds of thousands of features. The resulting anonymized datasets maintain sufficient structural fidelity to support accurate machine learning, statistical analysis, and data mining tasks, while ensuring that no individual user can be distinguished from at least k other users based on their pattern of feature associations.

## BACKGROUND

- motivate k-anonymity

The need to share user data for research, analytics, and commercial purposes while safeguarding individual privacy has driven the development of formal anonymization techniques over the past two decades. Among these, k-anonymity has emerged as a foundational and widely adopted principle, particularly in contexts where data must be released in its full form rather than as aggregated statistics. The core idea of k-anonymity is to ensure that every record in the published dataset is indistinguishable from at least k−1 other records when considering a specified set of quasi-identifying attributes. This prevents re-identification attacks by adversaries possessing auxiliary information about a target individual, as the target cannot be uniquely isolated within the dataset. In binary matrix representations, where rows correspond to users and columns to binary features, k-anonymity requires that each user’s feature vector must be identical to those of at least k−1 other users. Traditional approaches to achieving k-anonymity have relied primarily on suppression—removing or generalizing identifying features—which often results in significant loss of information, especially in sparse datasets where users have few feature associations. This leads to datasets that are either overly distorted or fail to meet the k-anonymity threshold without discarding a substantial fraction of the original data. Moreover, existing approximation algorithms for k-anonymity provide theoretical guarantees only under the assumption that the optimal solution requires removing only a vanishingly small fraction of edges, a condition rarely satisfied in real-world sparse data. In such cases, these algorithms may degenerate to trivial solutions, such as returning an empty dataset, which defeats the purpose of utility preservation. The limitations of suppression-based methods are further exacerbated in domains like social networks, web graphs, and user-interest profiles, where the underlying structure is inherently sparse and heterogeneous, and where even minor data loss can severely degrade the performance of downstream machine learning models. Consequently, there exists a critical gap between the theoretical guarantees of k-anonymity and its practical applicability in modern data ecosystems, motivating the development of a more robust and utility-aware anonymization framework.

## SUMMARY

- introduce k-anonymizing method

The present invention introduces a novel method for anonymizing binary datasets through a structured clustering and edge modification process that enforces smooth-k-anonymity, a relaxed yet more practical variant of traditional k-anonymity. Unlike prior methods that only suppress edges, this method permits both the addition and removal of feature-user associations under a majority-based constraint, ensuring that any new edge introduced in the anonymized dataset reflects a consensus among users within the same cluster. The method begins by representing each user as a binary vector in a high-dimensional space, where each dimension corresponds to a feature. These vectors are then clustered using a metric facility location algorithm that assigns users to clusters of size at least k, while minimizing the total Hamming distance between users and their assigned cluster centers. Each cluster is treated as an equivalence class, and for each feature, the system determines whether the majority of users in the cluster possess that feature. If so, all users in the cluster are assigned the feature; if not, all are stripped of it. This process ensures that the resulting dataset satisfies smooth-k-anonymity: every user is indistinguishable from at least k others, and any edge added to the output is supported by the majority of users in its cluster. The method guarantees a constant-factor approximation to the optimal smooth-k-anonymous solution under realistic conditions on data utility, and operates in polynomial time, making it scalable to datasets with billions of entries. The approach fundamentally improves upon suppression-based k-anonymity by preserving significantly more of the original data structure, thereby enhancing the utility of the anonymized dataset for analytical purposes without compromising privacy guarantees.

- describe clustering entities

The method employs a two-stage clustering process to group users into equivalence classes of minimum size k. In the first stage, each user is mapped to a point in an m-dimensional binary space, where m is the number of features, and the coordinates of the point correspond to the user’s feature vector. The Hamming distance between any two points is used as the metric to quantify dissimilarity between users. A facility location algorithm is then applied, wherein each user is treated as both a potential facility and a demand point, with an opening cost derived from the number of users within their k-nearest neighborhood. The algorithm selects a subset of these facilities to “open,” assigning each user to the nearest open facility such that the total cost—comprising both the sum of distances from users to their assigned facilities and the sum of opening costs—is minimized. This yields an initial clustering where each cluster contains at least αk users, for a tunable parameter α. In the second stage, clusters that are smaller than k are merged with adjacent clusters, subject to a constraint that no merged cluster exceeds 2k users. This merging process is performed iteratively, prioritizing pairs of clusters with the smallest inter-cluster distances, ensuring that the final clustering maintains a balance between cluster size and structural coherence. The resulting clusters are guaranteed to have sizes between k and 2k, which facilitates efficient majority computation and ensures that the anonymization process does not artificially homogenize overly large groups that may obscure meaningful substructure. This hierarchical clustering strategy is central to the invention’s ability to preserve data utility while satisfying the anonymity constraint.

- determine majority condition

Following the clustering of users, the invention enforces a majority condition to determine the final feature assignments within each cluster. For every feature and every cluster, the system evaluates the proportion of users within that cluster who originally possessed the feature. If this proportion exceeds 50%, the system assigns the feature to all users in the cluster; otherwise, it removes the feature from all users in the cluster. This majority-based decision rule ensures that the anonymized dataset does not introduce spurious associations that contradict the underlying data distribution. It also ensures that any edge added during anonymization is justified by consensus among the cluster members, thereby preventing the creation of misleading patterns that could be exploited by adversaries. The majority condition is applied independently across all features and clusters, making the process highly parallelizable and computationally efficient. Importantly, this condition distinguishes the invention from traditional k-anonymity by suppression, which only removes edges and cannot recover lost information. By allowing edge addition under majority control, the method preserves more of the original signal, particularly in sparse datasets where users have few features, and where suppression would otherwise obliterate meaningful patterns. The majority condition is mathematically proven to maintain the smooth-k-anonymity property while minimizing the Jaccard distance between the original and anonymized datasets, thereby optimizing utility under privacy constraints.

- assign data item to entities

Each user, represented as a binary vector of feature associations, is assigned a new vector in the anonymized dataset based on the cluster to which they are assigned. The assignment process is deterministic and follows directly from the majority condition applied to each cluster. For a given user u assigned to cluster c, the feature vector of u in the anonymized dataset is set equal to the consensus vector of cluster c. That is, for each feature f, if the majority of users in c originally had f, then u is assigned f; otherwise, u is assigned the absence of f. This assignment is applied uniformly to all users within the same cluster, ensuring that all users in cluster c have identical feature vectors in the output. The assignment does not depend on the original vector of the individual user beyond the cluster-level majority computation, thereby guaranteeing that no user can be uniquely identified from their anonymized vector alone. The process is lossless in the sense that no user is removed from the dataset, and the only modifications are the addition or removal of feature edges according to the majority rule. This ensures that the anonymized dataset retains the same number of users as the original, preserving the statistical power of the dataset for downstream analysis. The assignment step is the final transformation in the anonymization pipeline and directly produces the output dataset that satisfies smooth-k-anonymity.

- introduce computing system

The invention is implemented as a distributed computing system comprising multiple interconnected hardware components designed to process large-scale binary datasets efficiently. The system includes one or more central processing units, memory subsystems, and persistent storage devices capable of handling datasets exceeding one billion rows and one hundred thousand columns. The system further includes a network interface for receiving input datasets and distributing anonymized outputs to authorized users or downstream applications. The core anonymization algorithm is executed as a software module running on the central processing units, with memory allocated for storing the binary matrix, intermediate cluster assignments, and facility location data structures. The system is designed to operate in a single-threaded or multi-threaded mode depending on the scale of the input, and includes a parallelization module that partitions the dataset into chunks using locality-sensitive hashing to enable distributed processing across multiple nodes. The computing system is modular, allowing for integration with existing data pipelines, database systems, and machine learning platforms. It supports input in standard formats such as CSV, JSON, or sparse matrix representations, and outputs anonymized datasets in equivalent formats with metadata indicating the anonymization parameters, including k-value and cluster statistics.

- describe other aspects

The invention encompasses several additional aspects that enhance its practical utility and adaptability. First, the method supports dynamic adjustment of the k-anonymity parameter without reprocessing the entire dataset, enabling iterative privacy-utility trade-off exploration. Second, the system includes a utility evaluation module that computes the Jaccard similarity between the original and anonymized datasets, providing real-time feedback on the impact of anonymization. Third, the method can be extended to incorporate differential privacy guarantees by adding controlled noise to the cluster-level majority counts, creating a hybrid privacy model. Fourth, the system supports incremental updates, allowing new user records to be anonymized and integrated into an existing anonymized dataset without requiring full recomputation. Fifth, the invention includes a validation module that verifies the smooth-k-anonymity property of the output dataset, ensuring compliance with regulatory and ethical standards. Sixth, the system can be deployed in cloud environments, on-premise servers, or edge devices, making it suitable for use in diverse operational contexts ranging from academic research to commercial data sharing platforms. Finally, the method is agnostic to the semantic meaning of features, making it applicable across domains including healthcare, finance, advertising, and social network analysis.

## DETAILED DESCRIPTION

- introduce k-anonymity

K-anonymity is a privacy model that ensures that any individual record in a published dataset cannot be distinguished from at least k−1 other records when considering a specified set of attributes known as quasi-identifiers. In the context of binary datasets, where each row corresponds to a user and each column to a binary feature, k-anonymity requires that every user’s feature vector must appear at least k times in the dataset. This prevents re-identification attacks by adversaries who possess external knowledge about a target individual’s attributes, as the target cannot be uniquely isolated within the dataset. The model was originally proposed for tabular data and has since been adapted to graph-based representations where users and features form a bipartite structure. The primary challenge in achieving k-anonymity lies in the trade-off between privacy and utility: suppressing or generalizing features to create sufficient indistinguishability often results in the loss of meaningful patterns, particularly in sparse datasets where users have few feature associations. Traditional approaches to k-anonymity focus exclusively on suppression, which removes edges from the bipartite graph, but this leads to significant degradation in data quality. The present invention overcomes this limitation by introducing a new variant called smooth-k-anonymity, which allows for both edge addition and removal under a majority constraint, thereby preserving more of the original structure while still ensuring anonymity.

- motivate k-anonymity

The motivation for k-anonymity arises from the increasing demand to share user data for research, machine learning, and business intelligence purposes while protecting individual privacy. Regulatory frameworks such as HIPAA, GDPR, and CCPA require organizations to de-identify personal data before sharing, and k-anonymity provides a formal, interpretable mechanism to meet these requirements. Unlike differential privacy, which adds noise to outputs and often renders data unusable for fine-grained analysis, k-anonymity preserves the full structure of the dataset, enabling accurate statistical inference and model training. This is particularly critical in domains such as healthcare, where anonymized patient records must retain diagnostic patterns, or in advertising, where user interest profiles must remain predictive of behavior. Moreover, k-anonymity is computationally tractable and does not require complex probabilistic mechanisms, making it accessible to organizations without specialized privacy engineering expertise. The invention enhances the practical viability of k-anonymity by addressing its primary weakness—excessive data loss—through the introduction of smooth-k-anonymity, which enables more nuanced and utility-preserving transformations.

- describe differential privacy

Differential privacy is a formal privacy framework that guarantees that the output of a data analysis algorithm remains statistically indistinguishable whether or not any single individual’s data is included in the input. It achieves this by injecting calibrated random noise into the results, ensuring that an adversary cannot infer the presence or attributes of any specific individual. While differential privacy offers strong theoretical guarantees against arbitrary side information, it is often incompatible with the release of full datasets, as the noise required to protect individual records can obliterate the underlying signal, particularly in sparse binary data. For example, in a dataset where each user has only a few features, adding noise to preserve differential privacy may result in a randomized output that bears little resemblance to the original, rendering it useless for downstream tasks. Furthermore, achieving meaningful privacy levels often requires noise parameters that are impractically large, especially when the number of features is high. As a result, differential privacy is typically applied to aggregated statistics or learned models, not to raw data releases. The present invention provides an alternative that avoids noise injection entirely, instead relying on deterministic, structure-preserving transformations to achieve privacy, thereby offering superior utility in scenarios where full data release is required.

- limitations of differential privacy

The primary limitation of differential privacy in the context of binary matrix anonymization is its inability to preserve utility when the data is sparse. Theoretical analyses demonstrate that achieving even moderate Jaccard similarity between the original and differentially private output requires noise parameters that scale logarithmically with the number of features, which is infeasible for datasets with tens of thousands of dimensions. Empirical results show that for datasets with densities below 10⁻⁴, differential privacy mechanisms either produce outputs that are nearly random or require privacy parameters so large that they offer no meaningful protection. This renders differential privacy ineffective for real-world applications involving user-feature matrices, such as web graphs, co-authorship networks, or interest-based advertising profiles. Additionally, differential privacy mechanisms are often randomized, leading to non-reproducible outputs, which is undesirable in audit and compliance contexts. The invention overcomes these limitations by providing a deterministic, non-noise-based method that achieves comparable or superior utility while maintaining strong anonymity guarantees.

- introduce quasi-identifiers

In the context of k-anonymity, quasi-identifiers are a subset of attributes that, when combined, can be used to uniquely identify individuals when cross-referenced with external information. In traditional tabular data, quasi-identifiers might include age, zip code, and gender. In the binary matrix representation used by the invention, every feature is treated as a potential quasi-identifier, as any combination of features could potentially distinguish a user. This assumption is conservative and reflects the reality that in modern digital systems, even seemingly innocuous features—such as the apps installed on a device or the websites visited—can be combined to re-identify users with high accuracy. By treating all features as quasi-identifiers, the invention ensures that the anonymization process is robust against any possible auxiliary information an adversary might possess. This approach eliminates the need for manual selection of quasi-identifiers, which is error-prone and domain-specific, and instead applies a uniform anonymization strategy across all features.

- define k-anonymity

K-anonymity is formally defined as a property of a dataset in which every record is identical to at least k−1 other records with respect to the set of quasi-identifiers. In the binary matrix setting, this means that for every user u, there exist at least k−1 other users whose feature vectors are exactly equal to that of u. The goal of k-anonymization is to modify the dataset—typically by suppressing or generalizing values—so that this condition is satisfied. The resulting dataset ensures that an adversary observing the published data cannot determine which of the k individuals corresponds to a target, even with knowledge of their quasi-identifying attributes. The present invention satisfies this definition by grouping users into clusters of size at least k and assigning each user in a cluster the same feature vector, thereby ensuring that every user’s anonymized vector appears at least k times in the output.

- describe k-anonymity in terms of quasi-identifiers

In the context of the invention, quasi-identifiers are the binary features that define the user’s profile. K-anonymity is achieved when the set of features associated with any user is shared by at least k users in the anonymized dataset. The invention enforces this by ensuring that all users within a cluster have identical feature vectors, and that each cluster contains at least k users. Since every feature is treated as a quasi-identifier, the anonymization process modifies the entire feature vector of each user to match the cluster consensus, thereby rendering the user indistinguishable from the other k−1 members of the cluster. This approach eliminates the need to identify or select a subset of quasi-identifiers, as the method operates on the full feature space, ensuring comprehensive protection against all possible re-identification attacks based on feature combinations.

- introduce example aspects of the present disclosure

The present disclosure introduces a novel algorithmic framework for achieving smooth-k-anonymity in binary datasets through clustering and majority-based edge modification. Unlike prior methods that rely on suppression or noise injection, the invention permits both addition and removal of feature-user associations under a majority constraint, ensuring that any new association reflects a consensus among users in the same cluster. The method is implemented as a polynomial-time approximation algorithm with provable guarantees on utility preservation, and is designed to operate efficiently on sparse, high-dimensional datasets. The invention further includes a computing system architecture that enables scalable deployment across distributed environments, and supports integration with existing data infrastructure. The resulting anonymized datasets maintain high utility for machine learning and statistical analysis while providing strong, verifiable privacy guarantees.

- describe improved privacy guarantees

The invention provides improved privacy guarantees compared to traditional k-anonymity by suppression and differential privacy. By allowing edge addition under majority control, the method prevents adversaries from distinguishing between original and synthetic associations, thereby reducing the risk of inference attacks based on feature sparsity. Unlike suppression, which reveals which features were removed, the invention obscures the original data by making all users in a cluster identical, regardless of their initial profile. This ensures that an adversary cannot determine whether a missing feature was suppressed or never present. Furthermore, the majority condition ensures that no spurious associations are introduced, preventing the creation of misleading patterns that could be exploited. The method also avoids the noise-induced unreliability of differential privacy, producing deterministic, reproducible outputs that are suitable for audit and compliance. These combined properties result in a privacy model that is both stronger in practice and more usable in real-world applications.

- introduce computing system

The computing system of the invention comprises a hardware and software infrastructure designed to process large-scale binary datasets efficiently and securely. The system includes one or more central processing units, memory modules, and persistent storage devices capable of handling datasets with billions of rows and hundreds of thousands of columns. The anonymization algorithm is implemented as a software module that executes on the processing units, with memory allocated for storing the input matrix, intermediate cluster assignments, and facility location data structures. The system includes a network interface for receiving input datasets and transmitting anonymized outputs, and supports input in standard formats such as CSV, JSON, or sparse matrix encodings. The system is modular and can be deployed on-premise, in the cloud, or on edge devices, and includes a parallelization engine that partitions the dataset using locality-sensitive hashing to enable distributed processing across multiple nodes. The system also includes a validation module that verifies the smooth-k-anonymity property of the output and a utility evaluation module that computes the Jaccard similarity between the original and anonymized datasets.

- describe computing system components

The computing system consists of five core components: (1) a data ingestion module that accepts binary datasets in various formats and converts them into a standardized internal representation; (2) a clustering engine that maps users to points in a binary space and applies a metric facility location algorithm to group them into clusters of size between k and 2k; (3) a majority decision module that, for each cluster and feature, determines whether the majority of users originally had that feature and assigns the consensus value to all users in the cluster; (4) an output generation module that constructs the anonymized dataset by replacing each user’s feature vector with their cluster’s consensus vector; and (5) a validation and utility assessment module that verifies the smooth-k-anonymity property and computes the Jaccard similarity between the original and anonymized datasets. The system further includes a configuration interface that allows users to set parameters such as k, α, and parallelization settings, and a logging module that records anonymization decisions for audit purposes.

- introduce operations for k-anonymizing a dataset

The operations for k-anonymizing a dataset begin with the ingestion of a binary matrix where rows represent users and columns represent features. The system then embeds each user as a binary vector in an m-dimensional space, where m is the number of features. A metric facility location algorithm is applied to assign users to clusters such that each cluster contains at least k users and the total Hamming distance between users and their cluster centers is minimized. Clusters smaller than k are merged with adjacent clusters until all clusters have size between k and 2k. For each cluster and each feature, the system determines whether the majority of users in the cluster originally had that feature. If so, all users in the cluster are assigned the feature; otherwise, all are stripped of it. The resulting matrix is the anonymized dataset, which is then validated to ensure smooth-k-anonymity and evaluated for utility using the Jaccard similarity metric.

- obtain dataset

The system obtains a dataset through a data ingestion interface that accepts binary matrices in standard formats such as CSV, JSON, or sparse matrix encodings such as CSR or COO. The dataset may be provided via direct file upload, API endpoint, or secure data transfer protocol. The system validates the input to ensure it is a binary matrix with no missing values and that the number of rows and columns falls within system limits. The dataset is then loaded into memory as a sparse matrix representation to optimize storage and computation, particularly for datasets with low density.

- describe dataset

The dataset consists of a binary matrix with n rows and m columns, where each row corresponds to a user and each column to a feature. An entry of 1 indicates that the user possesses the feature, while a 0 indicates absence. The dataset may represent user-device associations, user-interest profiles, co-authorship networks, or web hyperlink structures. The dataset is typically sparse, with each user having only a small number of features, and the number of features may range from thousands to hundreds of thousands. The dataset may originate from user logs, survey responses, or behavioral tracking systems, and may contain up to one billion rows and ten billion non-zero entries.

- introduce clustering entities

Clustering entities refers to the process of grouping users into equivalence classes based on their feature similarity. Each user is treated as an entity in a high-dimensional binary space, and clustering is performed using a metric facility location algorithm that minimizes the total Hamming distance between users and their assigned cluster centers. The clustering process ensures that each cluster contains at least k users, and that the number of clusters is approximately n/k. The resulting clusters serve as the basis for the anonymization process, as all users within a cluster will have identical feature vectors in the output.

- describe clustering entities

The clustering of entities is performed using a two-stage algorithm. In the first stage, each user is represented as a point in an m-dimensional binary space, and a facility location problem is solved where each user is both a potential facility and a demand point. The opening cost of a facility is determined by the number of users within its k-nearest neighborhood. The algorithm selects a subset of facilities to open and assigns each user to the nearest open facility, minimizing the sum of distances and opening costs. This yields clusters of size at least αk. In the second stage, clusters smaller than k are iteratively merged with adjacent clusters, prioritizing those with the smallest inter-cluster distance, until all clusters have size between k and 2k. This ensures that clusters are neither too small to satisfy anonymity nor too large to obscure meaningful substructure.

- introduce mapping entities to points in dimensional space

Mapping entities to points in dimensional space involves representing each user as a binary vector in an m-dimensional space, where m is the number of features. Each dimension corresponds to a feature, and the value in that dimension is 1 if the user has the feature and 0 otherwise. This mapping transforms the problem of anonymizing a binary matrix into a geometric clustering problem, enabling the application of well-established algorithms from facility location and clustering theory. The Hamming distance between any two points is used as the metric to quantify dissimilarity between users.

- describe mapping entities to points

The mapping of entities to points is performed by converting each row of the binary matrix into a binary vector of length m, where m is the number of features. For example, if a user has features f₁, f₃, and f₅, their vector is [1,0,1,0,1,0,…,0]. This representation preserves all information in the original dataset and enables the use of geometric distance metrics such as Hamming distance. The mapping is deterministic and reversible, ensuring that the anonymization process does not lose information prior to clustering. The resulting point set is stored in memory as a sparse matrix to optimize computational efficiency.

- introduce establishing centers in dimensional space

Establishing centers in dimensional space involves selecting a subset of points from the user point set to serve as cluster centers. These centers are chosen to minimize the total cost of assigning users to clusters, where the cost includes both the distance from users to their assigned centers and the cost of opening a center. The centers are not required to be actual user points but may be any point in the space; however, the algorithm is designed to select centers from the user set to ensure interpretability and computational tractability.

- describe distributing entity clusters among centers

Distributing entity clusters among centers refers to the assignment of each user to the nearest open center, subject to the constraint that each center must serve at least k users. This assignment is computed using a facility location algorithm that balances the trade-off between minimizing distances and minimizing the number of centers opened. The algorithm iteratively selects centers and assigns users until all users are assigned and each cluster has at least k users. The resulting distribution ensures that clusters are compact and that the total distortion introduced by anonymization is minimized.

- introduce lower-bounded r-median approach

The lower-bounded r-median approach is a clustering technique that requires each cluster to contain at least k users while minimizing the total distance from users to their assigned cluster centers. The number of centers r is not fixed but is constrained by the total number of users and the minimum cluster size, such that r ≤ n/k. This approach is used in the first stage of the invention’s clustering process to produce an initial partitioning of users into clusters of size at least k.

- describe lower-bounded r-median approach

The lower-bounded r-median approach is implemented using a known 82.6-approximation algorithm that selects a set of centers and assigns users to them such that each center serves at least k users and the total Hamming distance is minimized. The algorithm begins by embedding users as points in a binary space and then applies a greedy selection process to identify candidate centers. It iteratively assigns users to the nearest center and enforces the k-user constraint by merging underfilled clusters. The result is a clustering where each cluster has size at least k, and the total distortion is within a constant factor of the optimal solution.

- introduce metric facility location approach

The metric facility location approach is a generalization of the lower-bounded r-median problem that introduces an opening cost for each potential center. The goal is to select a subset of centers to open and assign each user to an open center such that the sum of the opening costs and the assignment distances is minimized. This approach is used in the second stage of the invention’s clustering process to produce a more balanced and utility-preserving clustering.

- describe metric facility location approach

The metric facility location approach is implemented using a 1.488-approximation algorithm that treats each user as a potential facility with an opening cost proportional to the number of users within their k-nearest neighborhood. The algorithm selects a subset of facilities to open and assigns each user to the nearest open facility. The result is a clustering where clusters may initially exceed k users, but are later refined to ensure sizes between k and 2k. This approach provides better balance and lower total distortion than the lower-bounded r-median approach alone.

- introduce determining opening cost for each facility

The opening cost for each facility is determined based on the local density of users around that point. Specifically, for each user u, the opening cost is set to a function of the number of users within the k-nearest neighborhood of u. This ensures that facilities in dense regions are cheaper to open, encouraging the algorithm to form clusters in areas of high user similarity.

- describe opening cost calculation

The opening cost for a facility at user u is calculated as 2α / |U_k(u)|, where U_k(u) is the set of the k nearest users to u, and α is a tunable parameter. This formula ensures that facilities in sparse regions have higher opening costs, discouraging the algorithm from creating clusters in low-density areas. The cost function is designed to promote the formation of clusters where users are naturally similar, thereby preserving the underlying structure of the dataset.

- introduce assigning points to facilities

Assigning points to facilities involves determining, for each user, which open facility they are assigned to, such that the total cost of assignments and facility openings is minimized. This assignment is computed using a greedy algorithm that iteratively selects the nearest open facility for each user, subject to the constraint that no facility is assigned fewer than k users.

- describe assigning points to facilities

Each user is assigned to the open facility that minimizes the sum of the Hamming distance to the facility and the facility’s opening cost. The assignment is performed in a greedy manner, with users being processed in a random order to avoid bias. After all users are assigned, the algorithm checks that each facility serves at least k users; if not, the underfilled facility is closed, and its users are reassigned to the next closest open facility. This process continues until all clusters meet the minimum size requirement.

- introduce merging entity clusters

Merging entity clusters is the process of combining two or more clusters that are too small to satisfy the minimum k-user requirement. This step ensures that every cluster in the final anonymized dataset contains at least k users.

- describe merging entity clusters

Merging is performed iteratively, starting with the smallest clusters. For each underfilled cluster, the algorithm identifies the nearest other cluster and merges them, provided the resulting cluster does not exceed 2k users. The distance between clusters is measured as the average Hamming distance between users in one cluster and users in the other. This ensures that clusters are merged only with structurally similar neighbors, preserving data utility. The merging process continues until all clusters have size between k and 2k.

- introduce splitting entity clusters

Splitting entity clusters is an optional refinement step that may be applied to clusters larger than 2k to improve cluster homogeneity. This step is not required for the core anonymity guarantee but may be used to enhance utility in certain applications.

- describe splitting entity clusters

When a cluster exceeds 2k users, it is split into two sub-clusters using a k-means++ initialization approach, where the two new centers are selected to maximize the distance between them. Users are then reassigned to the nearest center, and the process is repeated if necessary. Splitting is performed only if it reduces the total intra-cluster Hamming distance, ensuring that utility is not compromised.

- introduce determining majority condition

The majority condition is the rule that determines whether a feature is assigned to all users in a cluster based on the proportion of users in the cluster who originally had that feature. This condition ensures that any edge added during anonymization reflects a consensus among cluster members.

- describe majority condition

For each cluster and each feature, the system counts the number of users in the cluster who originally had the feature. If this count exceeds half the cluster size, the feature is assigned to all users in the cluster; otherwise, it is removed from all users. This majority rule ensures that the anonymized dataset does not introduce spurious associations and that any added edge is justified by the original data distribution.

- introduce assigning data item to entities

Assigning data items to entities refers to the final step of replacing each user’s original feature vector with the consensus vector of their assigned cluster. This step produces the anonymized dataset.

- describe assigning data item to entities

Each user’s feature vector is replaced by the vector that results from applying the majority condition to their cluster. For example, if a cluster of 10 users has 7 users with feature f, then all 10 users are assigned feature f in the output. This assignment is deterministic and uniform across the cluster, ensuring that all users in the same cluster are indistinguishable. The assignment does not depend on the original vector of the individual user beyond the cluster-level majority computation, thereby guaranteeing anonymity.

- introduce anonymized dataset

The anonymized dataset is the output of the invention’s process, consisting of a binary matrix where each user’s feature vector has been replaced by the consensus vector of their cluster. This dataset satisfies smooth-k-anonymity: every user is indistinguishable from at least k others, and any feature assigned to a user is supported by the majority of users in their cluster.

- describe anonymized dataset

The anonymized dataset has the same number of rows and columns as the original dataset, but the feature vectors of users have been modified to reflect cluster-level consensus. The dataset is sparse, with the same or fewer non-zero entries than the original, and contains no user-specific patterns that could be used for re-identification. The dataset can be used for any downstream analytical task, including machine learning, statistical modeling, or data mining, with minimal loss of utility.

- introduce distributing anonymized dataset

Distributing the anonymized dataset involves transmitting the output to authorized users, researchers, or downstream systems in a secure and controlled manner.

- describe distributing anonymized dataset

The anonymized dataset is distributed via secure API endpoints, encrypted file transfer protocols, or access-controlled data repositories. Access is granted only to users with appropriate authorization, and the distribution includes metadata specifying the anonymization parameters, including k-value, cluster statistics, and utility metrics. The system logs all distribution events for audit and compliance purposes.

- introduce technical effects and benefits

The invention produces technical effects that include improved data utility, stronger privacy guarantees, and scalable anonymization of large-scale datasets. The method preserves significantly more of the original data structure than suppression-based k-anonymity or noise-based differential privacy, enabling accurate machine learning and statistical analysis. The deterministic nature of the output ensures reproducibility and auditability, while the polynomial-time algorithm enables deployment on datasets with billions of entries.

- describe technical effects and benefits

The technical benefits include a constant-factor approximation guarantee on utility preservation, linear scalability with dataset size, and compatibility with existing data infrastructure. The method reduces the number of suppressed features by up to 26% compared to state-of-the-art suppression methods, while increasing the number of added features by less than 10%, resulting in a net gain in utility. The anonymized datasets maintain high Jaccard similarity to the original, enabling accurate downstream analysis. The system is robust to sparsity and heterogeneous feature distributions, making it applicable across diverse domains.

- introduce improvements in computing technology

The invention improves computing technology by enabling efficient anonymization of massive binary datasets that were previously intractable under existing methods. The use of sparse matrix representations, locality-sensitive hashing, and parallelized clustering algorithms reduces memory usage and computation time by orders of magnitude.

- describe improvements in computing technology

The invention introduces novel algorithmic optimizations that reduce the computational complexity of clustering from O(n²m) to O(nm log n) through sparse representations and approximate facility location. The use of parallelization via locality-sensitive hashing enables processing of datasets with over one billion rows on commodity hardware. The system’s memory footprint is reduced by 80% compared to dense matrix approaches, making it feasible to run on edge devices and cloud instances with limited resources.

- introduce producing anonymized datasets

The invention provides a method for producing anonymized datasets that satisfy smooth-k-anonymity while preserving utility for analytical tasks.

- describe producing anonymized datasets

The method produces anonymized datasets by clustering users into groups of size k–2k and applying a majority-based feature assignment rule. The resulting datasets are deterministic, reproducible, and verifiable, and can be produced in polynomial time for datasets of any size. The method is applicable to any binary matrix representation of user-feature interactions and is not limited by the number of features or users.

- introduce applications of example aspects

The invention has applications in healthcare data sharing, advertising analytics, social network analysis, financial fraud detection, and government data publishing.

- describe applications of example aspects

In healthcare, the invention enables sharing of patient symptom profiles without revealing individual identities. In advertising, it allows sharing of user interest profiles for targeted marketing while protecting privacy. In social networks, it permits release of co-authorship or friendship graphs without exposing individual identities. In finance, it enables anonymized transaction pattern analysis for fraud detection. In government, it supports release of census or survey data that meets regulatory privacy requirements.

- introduce computing systems

The invention is implemented as a computing system that includes hardware, software, and network components designed to process and distribute anonymized datasets.

- describe first computing system

The first computing system is a single-node server with high-memory capacity, designed for processing datasets up to one hundred million rows. It includes a multi-core processor, 512 GB of RAM, and 10 TB of SSD storage. The anonymization algorithm runs as a single-threaded process with optimized memory access patterns to minimize latency.

- describe second computing system

The second computing system is a distributed cluster of 100 nodes, each with 64 GB of RAM and 2 TB of storage, designed for processing datasets exceeding one billion rows. The system uses a master-worker architecture, where the master distributes data chunks via locality-sensitive hashing and coordinates cluster merging across workers. The system achieves linear scalability and can anonymize a one-billion-row dataset in under four hours.

- describe network

The network connects the computing system to data sources and users, and includes firewalls, encryption protocols, and access control mechanisms to ensure secure data transmission. Data is transmitted over TLS 1.3, and access is granted only to authenticated users with appropriate permissions.

- introduce data anonymization

Data anonymization is the process of modifying a dataset to prevent the identification of individuals while preserving its utility for analysis.

- describe example graphs

Example graphs include bipartite graphs representing user-feature interactions, such as user-app installations, user-interest subscriptions, or co-authorship networks. These graphs are sparse, with each user connected to only a few features.

- describe k-anonymized graph

A k-anonymized graph is a bipartite graph where each user’s feature vector is identical to at least k−1 other users’ vectors. This is achieved by suppressing edges to create equivalence classes of size k.

- describe k-smooth-anonymized graph

A k-smooth-anonymized graph is a bipartite graph where each user’s feature vector is identical to at least k−1 others, and any edge added to the graph is supported by the majority of users in the same cluster. This ensures that the graph preserves more of the original structure than a k-anonymized graph.

- introduce flow chart diagram

The invention includes a flow chart diagram illustrating the steps of the anonymization process, from dataset ingestion to output distribution.

- describe obtaining dataset

The flow chart begins with the ingestion of a binary matrix from a file or API, followed by validation and conversion to a sparse representation.

- describe dataset formats

The system accepts datasets in CSV, JSON, or sparse matrix formats such as CSR, COO, or DOK. The input must be binary, with no missing values.

- describe clustering entities

The flow chart shows the mapping of users to points, followed by the application of the metric facility location algorithm to form clusters.

- describe mapping entities to points

Each user is converted into a binary vector of length m, where m is the number of features, and stored as a sparse array.

- describe establishing centers

The algorithm selects candidate centers based on local density and computes opening costs for each.

- describe distributing entities among centers

Users are assigned to the nearest open center, subject to the k-user constraint.

- describe lower-bounded r-median approach

The system applies a known approximation algorithm to solve the lower-bounded r-median problem.

- describe metric facility location approach

The system applies a 1.488-approximation algorithm to solve the metric facility location problem.

- describe opening cost calculation

Opening costs are computed as 2α divided by the number of users in the k-nearest neighborhood.

- describe assigning points to facilities

Each user is assigned to the open facility that minimizes the sum of distance and opening cost.

- describe merging entity clusters

Underfilled clusters are merged with their nearest neighbors until all clusters have size between k and 2k.

- describe splitting entity clusters

Clusters larger than 2k are optionally split into two sub-clusters using k-means++ initialization.

- describe determining majority condition

For each cluster and feature, the system counts the number of users with the feature and applies the majority rule.

- describe assigning data items

Each user’s feature vector is replaced by the cluster consensus vector.

- describe anonymized dataset

The output is a binary matrix with the same dimensions as the input, but with modified feature vectors.

- describe k-smooth-anonymized dataset

The output dataset satisfies smooth-k-anonymity: each user is indistinguishable from at least k others, and all added edges are supported by majority.

- describe distributing anonymized dataset

The anonymized dataset is transmitted via secure API or encrypted file to authorized users.

- describe protecting entities from privacy violations

The method protects entities by ensuring that no user can be uniquely identified from their anonymized feature vector, and that any inference about an individual’s features is indistinguishable from inference about k−1 other individuals.

- introduce flexibility of computer-based systems

The invention is implemented as a flexible software system that can be deployed on a variety of hardware configurations, from single servers to distributed clusters.

- describe single device or component implementation

The system can be implemented on a single server with sufficient memory to handle datasets up to 100 million rows.

- describe multiple devices or components implementation

The system can be distributed across multiple nodes, with data partitioned using locality-sensitive hashing and processed in parallel.

- describe distributed components operation

Distributed components operate independently but coordinate via a master node that aggregates cluster assignments and performs final merging.

- describe databases and applications implementation

The system integrates with relational and NoSQL databases, and can be embedded as a library in machine learning pipelines.

- describe alterations to embodiments

Alterations to the embodiments include changing the value of k, adjusting the α parameter, or modifying the clustering algorithm to use alternative distance metrics.

- describe variations of embodiments

Variations include using weighted features, incorporating temporal dynamics, or extending the method to categorical data.

- describe equivalents to embodiments

Equivalents include using different approximation algorithms for facility location or replacing Hamming distance with Jaccard distance.

- describe inclusion of modifications

Modifications may include adding differential privacy noise to cluster counts, or integrating with differential privacy frameworks for hybrid guarantees.

- describe features of one embodiment with another

Features from the distributed implementation may be combined with the single-device version to create a hybrid system that scales adaptively.

- describe alterations to embodiments

Alterations may include changing the upper bound on cluster size from 2k to 3k, or modifying the opening cost function to include feature entropy.

- describe variations of embodiments

Variations include applying the method to directed graphs, or extending it to multi-label classification tasks.

- describe equivalents to embodiments

Equivalents include implementing the algorithm in a functional programming language or using quantum computing for clustering optimization.

- describe inclusion of modifications

Modifications may include adding user-level privacy budgets or integrating with federated learning frameworks.

- describe features of one embodiment with another

The majority condition from the clustering embodiment may be combined with the parallelization strategy from the distributed embodiment to create a scalable, utility-preserving anonymizer.