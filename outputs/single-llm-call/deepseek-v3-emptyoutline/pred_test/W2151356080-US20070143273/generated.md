Here is the drafted patent application following the provided outline:

# DESCRIPTION  

## BACKGROUND OF THE INVENTION  

### Comparison of Information Retrieval Systems of MEDLINE  

Current information retrieval systems for MEDLINE, including PubMed, OVID, and other commercial search engines, suffer from significant limitations in identifying relevant biomedical literature. While these systems effectively retrieve articles containing all user-specified query terms, they fail to distinguish whether the retrieved articles actually discuss meaningful relationships between those terms. This fundamental shortcoming results in users being inundated with numerous irrelevant articles that happen to contain the specified keywords without establishing substantive connections between them.  

Existing systems primarily rely on article-level keyword matching without incorporating linguistic analysis of how terms relate within the text. For instance, when searching for "infection" and "sudden infant death syndrome" (SIDS), current systems retrieve all articles mentioning both terms anywhere in the text, regardless of whether the article actually discusses any causal or correlational relationship between infections and SIDS. This leads to inefficient literature searches where users must manually review numerous irrelevant articles to find those discussing the desired conceptual relationships.  

### Estimating Number of Words per Query in Queries Submitted to NLM's PubMed  

Analysis of query patterns submitted to PubMed reveals that approximately 83% of searches consist of multi-word queries. This high percentage demonstrates that most users are searching for articles discussing relationships between concepts rather than simply articles containing isolated terms. However, current search systems treat multi-word queries as simple Boolean AND operations, retrieving articles where all terms appear without considering their contextual relationships.  

The prevalence of multi-word queries underscores the need for a search system that can detect and prioritize articles where query terms appear in meaningful relationships. Statistical analysis shows that when two terms appear within the same sentence, there is a significantly higher probability (approximately 87% based on linguistic studies) that the article discusses a substantive relationship between those concepts compared to when terms appear in separate sentences or sections of an article.  

## SUMMARY OF THE INVENTION  

The present invention provides a novel information retrieval system for MEDLINE that substantially improves search precision by incorporating sentence-level co-occurrence analysis into relevance scoring. The system comprises three main components:  

1) A pre-processing component that parses MEDLINE records into discrete sentences and establishes a searchable database structure optimized for sentence-level retrieval;  

2) A user interface that accepts natural language queries and presents results with highlighted term matches within their sentence context;  

3) A search engine that implements an eight-tiered relevance scoring system based on term co-occurrence patterns across title, abstract, and MeSH fields.  

The system achieves superior precision compared to conventional search engines by prioritizing articles where query terms appear in close proximity within sentences, while maintaining high recall through comprehensive indexing of all MEDLINE content. Results are dynamically sorted by relevance score, ensuring users see the most pertinent articles first.  

## DESCRIPTION OF THE DRAWINGS  

Figure 1 illustrates the system architecture showing the flow from MEDLINE XML data through preprocessing to the searchable database and user interface components.  

Figure 2 depicts the precision comparison between the present invention and PubMed for the SIDS and infection case study, demonstrating significantly higher initial precision with the invented system.  

Figure 3 shows the user interface displaying search results with highlighted term matches within their sentence context and the associated relevance level indicators.  

## DESCRIPTION OF THE INVENTION  

### The Pre-Processing Component  

The pre-processing component transforms raw MEDLINE XML records into an optimized database structure for sentence-level searching. The system first extracts title, abstract, and MeSH terms from each article record. Using advanced sentence boundary detection algorithms, it then segments abstract text into discrete sentences while handling special cases such as abbreviations (e.g., "et al.") and numeric expressions (e.g., "p < 0.05") that might otherwise cause incorrect sentence splitting.  

The processed data is stored in a relational database with two primary tables:  
1) A sentences table containing each parsed sentence along with metadata including PubMed ID (PMID) and sentence sequence number;  
2) A citations table containing bibliographic information linked to sentences through PMIDs.  

The database implements specialized indexing to enable rapid sentence-level searches while maintaining connections to full article metadata. The pre-processing component also incorporates Unified Medical Language System (UMLS) mappings to expand terms with their synonyms and related concepts during indexing.  

### The User Interface  

The user interface accepts natural language queries through a web-based front-end. Users may enter single or multi-word queries, with support for Boolean operators (AND, OR, NOT), phrase matching (using quotation marks), and term truncation (using asterisks). The interface automatically suggests term expansions based on UMLS mappings to improve recall.  

Search results are presented with each matching sentence shown in context, with query terms highlighted for easy identification. The interface displays the article title, journal information, and publication date for each result, along with a visual indicator of the relevance level (1-8). Users can click through to view the complete article in PubMed while maintaining their sorted result order.  

### The Search Engine  

The search engine implements a sophisticated relevance scoring algorithm based on eight defined levels of term co-occurrence:  

Level 1 (Highest Relevance): Query terms appear in title AND in at least one abstract sentence AND in MeSH terms  
Level 2: Terms in title AND at least one abstract sentence  
Level 3: Terms in title AND MeSH terms  
Level 4: Terms in at least one abstract sentence AND MeSH terms  
Level 5: Terms in title only  
Level 6: Terms in MeSH only  
Level 7: Terms in abstract only (but same sentence)  
Level 8 (Lowest Relevance): Terms anywhere in article (different sentences)  

The engine first retrieves all articles containing the query terms, then assigns each article to the highest matching relevance level. Results are sorted by relevance level, with articles in the same level sorted chronologically. This approach ensures users see articles most likely to discuss relationships between query terms first, while still having access to all potentially relevant literature.  

## EXAMPLE 1  

### Role of 'Infection' in 'Sudden Infant Death Syndrome' (SIDS)  

A search for "sids (infection or infect*)" demonstrates the system's advantages. Conventional search returns 608 articles with both terms present somewhere in the text. The invented system retrieves 927 articles but sorts them such that the 32 articles discussing infection as a potential cause of SIDS appear first (Relevance Level 1). These articles contain phrases like "respiratory infections may predispose infants to SIDS" in their titles and abstracts.  

Precision analysis shows the first 10 results from the invented system have 100% relevance (all discussing infection-SIDS relationships), compared to 30% for conventional search. This 3-fold improvement in initial precision significantly reduces time spent reviewing irrelevant articles.  

## EXAMPLE 2  

### Finding 'Questionnaires' for Measuring 'Health Literacy'  

A search for health literacy assessment tools demonstrates the system's ability to surface methodological literature. The query ""health literacy" and (instrument* or question*)" returns 158 articles. The invented system prioritizes 14 articles (Relevance Level 1) that specifically describe health literacy questionnaires in their titles and abstracts, such as "Development and validation of the Health Literacy Questionnaire (HLQ)."  

Conventional search returns these relevant articles scattered throughout results, forcing users to review 39 irrelevant articles first. The invented system's precision of 68.9% in the first 28 results doubles conventional search's 39.3%, demonstrating more efficient access to target literature.  

## The Distributed Parallel Computing Architecture  

The system implements a distributed architecture using Linux, Apache, MySQL, and Perl (LAMP stack) to enable scalable processing of MEDLINE's growing content. The database shards sentence data across multiple servers to maintain performance as the corpus expands. Query processing distributes search operations across nodes, with results aggregated and sorted by a central coordinator.  

This architecture supports rapid updates to incorporate new MEDLINE records while maintaining sub-second response times for complex queries. The system can scale horizontally by adding additional processing nodes as needed to handle increasing query loads or database size.