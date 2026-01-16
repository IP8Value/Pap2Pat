# DESCRIPTION

## STATEMENT REGARDING PRIOR DISCLOSURES BY THE INVENTOR OR JOINT INVENTOR

The inventors have previously disclosed aspects of the invention in a research paper titled "Efficient Privacy-Preserving Record Linkage Using Locality-Sensitive Hashing and Private Set Intersection." The disclosure was made on March 27, 2022, and is available on arXiv with the identifier arXiv:2203.14284v1 [cs.CR].

## BACKGROUND

Entity resolution (ER) is a critical process in data integration, particularly when multiple datasets from different sources need to be combined. The primary goal of ER is to identify and link records that refer to the same entity across different datasets. However, when these datasets contain sensitive information, such as medical, financial, or criminal records, privacy concerns arise. Organizations are often legally restricted from sharing such sensitive data, making traditional ER methods infeasible.

The problem of matching records in two or more datasets without revealing additional information is known as privacy-preserving record linkage (PPRL) or blind data linkage (BDL). PPRL is a generalization of the private set intersection (PSI) problem, where two parties with different datasets seek to determine the intersection of their datasets without revealing any additional information. While PSI focuses on exact matches, PPRL extends this to handle non-exact matches, which are common in real-world datasets due to variations in data entry, typos, and other discrepancies.

Existing PPRL solutions often involve third parties or do not provide thorough leakage analysis, which can compromise the security and privacy of the data. There is a need for an efficient and secure PPRL solution that can handle large datasets and provide strong privacy guarantees.

## SUMMARY

The present invention provides a novel and efficient privacy-preserving record linkage (PPRL) protocol that combines locality-sensitive hashing (LSH) and private set intersection (PSI). The protocol allows two parties to identify similar records in their respective datasets without revealing any additional information about the non-matched records. The invention is particularly useful for scenarios where datasets contain sensitive information and direct data sharing is prohibited.

Key features of the invention include:
- **Combination of LSH and PSI**: The protocol uses LSH to handle non-exact matches and PSI to ensure privacy.
- **Linear Time Complexity**: The protocol runs in O(n) time, making it suitable for large datasets.
- **Security Against Semi-Honest Adversaries**: The protocol is designed to be secure against semi-honest adversaries, ensuring that no additional information is revealed beyond the matched records.
- **No Third Parties**: The protocol does not require the involvement of a third party, enhancing security and privacy.
- **Optimizations**: The invention includes several lower-level and higher-level optimizations to improve performance and reduce computational overhead.

The protocol consists of the following steps:
1. **Preprocessing**: Both parties preprocess their datasets to extract relevant fields and apply LSH to generate band signatures.
2. **PSI Execution**: The band signatures are used as inputs to a PSI protocol, which identifies the intersecting signatures.
3. **Mapping Back**: The intersecting signatures are mapped back to the original records to identify the matched records.

The invention also provides a formal definition of PPRL and discusses various variants, including mutual PPRL, N-PPRL, and revealing PPRL. The protocol is evaluated over a dataset with 2^20 records, demonstrating its practical advantage with execution times ranging from 11 to 45 minutes, depending on network settings.

## DETAILED DESCRIPTION

### Introduction

The invention addresses the problem of privacy-preserving record linkage (PPRL) by combining locality-sensitive hashing (LSH) and private set intersection (PSI). The goal is to enable two parties to identify similar records in their respective datasets without revealing any additional information about the non-matched records. This is particularly important when dealing with sensitive data, such as medical, financial, or criminal records, where direct data sharing is prohibited by law.

### Problem Statement

Traditional entity resolution (ER) methods are effective for identifying similar records in datasets but often require direct data sharing, which can compromise privacy. Privacy-preserving record linkage (PPRL) aims to solve this problem by allowing parties to identify similar records without revealing additional information. However, existing PPRL solutions often involve third parties or do not provide thorough leakage analysis, which can compromise security and privacy.

### Solution Overview

The invention introduces a novel PPRL protocol that combines LSH and PSI. The protocol consists of the following steps:
1. **Preprocessing**: Both parties preprocess their datasets to extract relevant fields and apply LSH to generate band signatures.
2. **PSI Execution**: The band signatures are used as inputs to a PSI protocol, which identifies the intersecting signatures.
3. **Mapping Back**: The intersecting signatures are mapped back to the original records to identify the matched records.

### Detailed Protocol Description

#### Preprocessing

Both parties, \( P_s \) and \( P_r \), hold datasets \( D_s \) and \( D_r \) of sizes \( N_s \) and \( N_r \), respectively. Each dataset contains records with fields that can be used for matching, such as first name, last name, address, and date of birth. The preprocessing step involves the following:
1. **Field Extraction**: Extract the relevant fields from each record.
2. **Normalization**: Normalize the extracted fields to ensure consistency (e.g., converting text to lowercase, removing punctuation).
3. **LSH Application**: Apply LSH to the normalized fields to generate band signatures. The LSH function is designed to hash similar inputs to the same output hash value, allowing for non-exact matches.

The LSH function is based on the Jaccard index and Min-Hash. The Jaccard index measures the similarity of two sets by dividing the size of the intersection by the size of the union. Min-Hash is used to approximate the Jaccard index efficiently. The LSH output is a tuple of band signatures, where each band signature is a hash of a subset of the Min-Hash values.

#### PSI Execution

The band signatures generated in the preprocessing step are used as inputs to a PSI protocol. The PSI protocol allows the two parties to compute the intersection of their band signatures without revealing any additional information. The PSI protocol used in this invention is based on the Diffie-Hellman (DH) key agreement scheme, which is known for its simplicity and security.

The PSI protocol proceeds as follows:
1. **Key Generation**: Both parties generate their own secret keys \( sk_s \) and \( sk_r \).
2. **Encryption and Exchange**: Each party encrypts their band signatures using their secret key and sends the encrypted signatures to the other party.
3. **Intersection Computation**: The receiving party raises the received encrypted signatures to the power of their own secret key and sends the results back to the original party.
4. **Decryption and Intersection**: The original party decrypts the received values and identifies the intersecting signatures.

#### Mapping Back

The intersecting signatures are mapped back to the original records to identify the matched records. This step involves the following:
1. **Signature Matching**: Identify the band signatures that appear in both sets.
2. **Record Identification**: Map the intersecting band signatures back to the original records to determine the matched records.

### Security Analysis

The security of the protocol is analyzed against semi-honest adversaries, where all parties follow the protocol but may try to infer additional information from the intermediate computations and messages. The protocol ensures that:
- **Privacy of \( P_s \)**: \( P_s \) only learns the set of matched records and the size of \( D_r \).
- **Privacy of \( P_r \)**: \( P_r \) only learns the size of \( D_s \).

The security of the protocol is guaranteed by the one-way property of the hash function, the computational hardness of the decisional Diffie-Hellman (DDH) problem, and the one-more-Diffie-Hellman (OMDH) assumption. The DDH problem is used to hide the data in transit from eavesdroppers, while the OMDH assumption is used to prevent \( P_s \) from generating new records in the name of \( P_r \).

### Variants of the Protocol

#### Mutual PPRL

A mutual PPRL protocol allows both parties to learn the matched records. This is achieved by modifying the PSI step to use a mutual DH-PSI protocol. The security of the mutual PPRL protocol follows from the security of the mutual DH-PSI protocol or from the fact that the mutual protocol is equivalent to running the original PPRL protocol twice.

#### N-PPRL

An N-PPRL protocol allows the parties to learn the number of matches without revealing the identity of the matched records. This is achieved by reordering the encrypted band signatures during the PSI step in a way that hides the identity of the matched records but still enables them to be counted.

#### Revealing PPRL

A revealing PPRL protocol allows the parties to learn the matched records and the field content of the matched records. This is achieved by adding an additional step where the parties perform a PSI over the shingles of the matched records to compute the Jaccard index.

### Performance Evaluation

The protocol was evaluated over a dataset with 2^20 records. The experiments were conducted on two machines located in different local area networks (LANs) with an average round-trip latency of 65 ms. The machines had the following specifications:
- **Machine A**: Intel® Xeon® CPU E5-2620 v3 @ 2.40GHz, 12 physical cores, 377 GB of RAM.
- **Machine B**: Intel® Xeon® CPU E5-2699 v4 @ 2.20GHz, 44 physical cores, 744 GB of RAM.

The protocol was implemented in C++ and used OpenSSL version 1.1.1f for secure TLS 1.3 connections. The experiments showed that the protocol could handle large datasets efficiently, with execution times ranging from 11 to 45 minutes, depending on network settings.

### Conclusion

The invention provides a novel and efficient PPRL protocol that combines LSH and PSI to handle non-exact matches while ensuring strong privacy guarantees. The protocol is secure against semi-honest adversaries and does not require the involvement of a third party. The protocol has been evaluated over a large dataset, demonstrating its practical advantage in terms of performance and security. The implementation is freely available for testing, making it a valuable tool for organizations that need to perform PPRL on sensitive data.