# DESCRIPTION

## BACKGROUND OF THE INVENTION

### Comparison of Information Retrieval Systems of MEDLINE

The U.S. National Library of Medicine's (NLM) primary literature database, MEDLINE, indexes over 15 million citations in various fields including medicine, nursing, dentistry, veterinary medicine, the healthcare system, and preclinical sciences. Despite the extensive indexing, encountering extraneous articles in response to a query is not uncommon. While every retrieved article contains all the query words, the mere presence of these words does not guarantee the article's relevance to the user's query. This highlights a critical issue: the presence of query words in an article is a necessary but not a sufficient condition for relevance.

About 83% of queries submitted to PubMed, NLM's search engine for MEDLINE, are multi-word queries. Users typically seek a specific relationship between the query words, making the presence of such a relationship a necessary condition for relevance. Various methods exist to detect the presence and type of relationships between words in a text, but none of the existing information retrieval systems for MEDLINE incorporates these methods. As the size of the database grows, eliminating irrelevant articles without missing relevant ones becomes increasingly challenging. Traditional methods that increase specificity often sacrifice sensitivity, leading to a higher likelihood of missing relevant articles.

### Estimating Number of Words per Query in Queries Submitted to NLM's PubMed

The vast majority of queries submitted to PubMed are multi-word queries, indicating that users are often searching for specific relationships between multiple terms. This necessitates a more sophisticated approach to information retrieval that goes beyond simple keyword matching. The challenge lies in developing a system that can accurately detect and prioritize articles that not only contain the query words but also explain the relationships between them.

## SUMMARY OF THE INVENTION

The present invention provides a novel search engine for MEDLINE that enhances the precision and efficiency of information retrieval. The search engine, named Relemed, retrieves relevant articles by detecting sentence-level concurrence of search terms. It estimates a relevance score where the presence of a relationship between the query words is a crucial component. This approach ensures that the most relevant articles are displayed at the top of the search results, thereby reducing the user's time and effort in identifying pertinent information.

Relemed achieves this by pre-processing MEDLINE data, extracting title, abstract, and citation information, and indexing sentences. The system then matches user queries to these indexed sentences, prioritizing those where the query words co-occur within the same sentence. The relevance score is calculated based on the importance of different sentence types (title, abstract, MeSH terms) and the level of concurrence between the query words. This method significantly improves the specificity of search results without compromising sensitivity.

## DESCRIPTION OF THE DRAWINGS

Fig. 1: Schematic representation of the Relemed system architecture, including data preprocessing, database design, and user interface components.
Fig. 2: Precision trends for Relemed and PubMed in Case Study 1: Role of 'Infection' in 'Sudden Infant Death Syndrome' (SIDS).
Fig. 3: Precision trends for Relemed and PubMed in Case Study 2: Finding 'Questionnaires' for Measuring 'Health Literacy'.

## DESCRIPTION OF THE INVENTION

### The Pre-Processing Component

The Relemed system begins by obtaining MEDLINE data in extensible markup language (XML) format through a lease contract with NLM. The data is pre-processed to extract title, abstract, and citation information from each article record. The abstract text is then scanned to detect and separate sentences using delimiters such as '.', '?', and '!'. Consecutive sentences are rejoined where the period is sandwiched by single capital letters, specific words (e.g., 'etc.', 'et al.'), or digits (e.g., '0.05').

A database is designed with two tables to store the processed data. The first table contains the sentences, with an index created for efficient querying. Each sentence is assigned a unique identifier (SNTNCID), with the title being assigned SNTNCID 1. The second table stores citation information such as author names, article title, journal name, publication date, and page numbers. A many-to-one relationship exists between the two tables, allowing for the retrieval of citation information based on the PMID (PubMed ID).

### The User Interface

The user interface of Relemed is designed to be intuitive and user-friendly. Users can submit queries composed of one or more words, separated by spaces. By default, the system uses the Boolean 'and' operator to connect the words, but it also supports 'or' and 'not' operators. Truncation using the asterisk (*) and exact phrase matching using quotes ("") are also supported, aligning with PubMed's query language.

Upon receiving a query, the system prepares the query in SQL (Structured Query Language), interrogates the database, formats the results in HTML (HyperText Markup Language), and posts them back to the user's browser. The results are displayed in a structured format, with each matching sentence highlighted and accompanied by the publication information of the corresponding article. A hyperlink is provided for easy navigation to the respective PubMed article for further exploration.

### The Search Engine

The core of the Relemed system is its search engine, which retrieves articles by detecting sentence-level concurrence of search terms. The system assigns importance weights to different sentence types (title, abstract, MeSH terms) and combines them to define several levels of relevance. This allows the system to measure how closely an article answers the user's query and sort the results accordingly.

The relevance metric is defined across eight levels, with the most stringent criteria at the highest level. For example, at relevance level one, both query words must appear in the title, at least one sentence in the abstract, and in the MeSH terms. This ensures that the matched article is highly likely to be relevant to the user's query. The lower relevance levels gradually relax the criteria, allowing for a broader search while maintaining a balance between sensitivity and specificity.

## EXAMPLE 1

### Role of ‘Infection’ in ‘Sudden Infant Death Syndrome’ (SIDS)

Sudden Infant Death Syndrome (SIDS) is the unexplained death of an infant under one year old. Despite extensive research, no definitive cause has been identified, but various factors have been proposed, including recent infection. In this example, the user aims to retrieve articles that link infection as a potential cause of SIDS or explain the absence of such a relationship.

Using the query 'sids (infection or infect*)' in both PubMed and Relemed, the results were compared. PubMed returned 608 articles, while Relemed returned 927. The discrepancy in the number of articles is attributed to differences in the databases and the handling of synonyms and truncated words. Relemed's ability to detect sentence-level concurrence and assign relevance scores resulted in a higher precision at the start of the results. The precision trend in Relemed started at 100% and decreased gradually, while PubMed's precision varied more erratically.

## EXAMPLE 2

### Finding ‘Questionnaires’ for Measuring ‘Health Literacy’

Health literacy refers to the capacity of individuals to obtain, process, and understand basic health information and services needed to make appropriate health decisions. In this example, the user seeks publications that provide information about existing questionnaires or instruments for measuring health literacy.

Using the query "health literacy" and (instrument* or question* or measur* or scale* or assessment* or index* or test*), PubMed returned 157 articles, while Relemed returned 158. The overlap between the two sets of results was 96.8%, with Relemed identifying a few additional relevant articles. The precision trend in Relemed started at 100% and decreased gradually, demonstrating the effectiveness of the sentence-level concurrence and relevance scoring.

## The Distributed Parallel Computing Architecture

To handle the large volume of data and ensure fast response times, Relemed employs a distributed parallel computing architecture. The system is built using open-source software, including Perl for data preprocessing and query application, MySQL for the database, and Apache for serving HTTP requests. The server runs on a Fedora operating system, forming the LAMP (Linux Apache MySQL Perl) architecture. XHTML is used to produce the user interface and reports, ensuring compatibility and accessibility.

The distributed architecture allows the system to efficiently process and index the vast amount of MEDLINE data, enabling quick and accurate retrieval of relevant articles. The parallel processing capabilities ensure that the system can handle multiple queries simultaneously, providing a seamless user experience.

In conclusion, Relemed represents a significant advancement in information retrieval systems for MEDLINE. By focusing on sentence-level concurrence and incorporating a relevance metric, Relemed delivers highly relevant search results, enhancing the precision and efficiency of the search process. Further evaluation and comparison with other search engines will continue to refine and improve the system, making it an invaluable tool for researchers and healthcare professionals.