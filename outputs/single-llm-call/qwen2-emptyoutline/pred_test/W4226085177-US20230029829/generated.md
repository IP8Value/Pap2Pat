# DESCRIPTION

## BACKGROUND

In the realm of information retrieval and knowledge discovery, the ability to efficiently and accurately retrieve relevant information is paramount. Traditional ad-hoc retrieval systems, which typically respond to a single query with a list of documents, often fall short in providing the nuanced and detailed insights that users require. The advent of conversational search paradigms has addressed this limitation by enabling a more interactive and iterative process of information seeking. Conversational search systems engage users in a dialogue, allowing them to refine their queries and receive more targeted and relevant responses.

A critical component of conversational search is the mechanism for generating or selecting clarification questions. These questions help the system better understand the user's intent and provide more accurate and useful information. Existing approaches for generating clarification questions can be broadly categorized into two types: selection and generation. The selection approach involves choosing clarification questions from a predefined pool, while the generation approach involves dynamically creating new questions using rule-based or neural generative models.

While the generation approach offers a more flexible and contextually rich interaction, the selection approach provides a more controlled and less noisy environment. This is particularly useful in scenarios where the pool of clarification questions can be curated from reliable sources, such as query logs or historical interactions. The selection approach also benefits from the ability to leverage pre-existing knowledge and reduce the computational complexity associated with generating new questions.

This invention focuses on a method for selecting clarification questions in content-grounded conversations. Content-grounded conversations are those that start with an initial user query, proceed through several rounds of dialogue, and conclude with the retrieval of one or more documents. The system aims to predict the next clarification question based on the conversation context, using a combination of deep learning models and passage retrieval techniques.

## SUMMARY

The present invention provides a method and system for selecting clarification questions in content-grounded conversational search scenarios. The method involves the following steps:

1. **Conversation Context Analysis**: The system analyzes the conversation context, which includes the initial user query and subsequent dialogue utterances, to understand the user's information needs and the current state of the conversation.

2. **Passage Retrieval**: Based on the conversation context, the system retrieves relevant passages from a corpus of documents. These passages are used to provide additional context and information that can help in formulating appropriate clarification questions.

3. **Clarification Question Retrieval**: The system uses the retrieved passages to identify a set of candidate clarification questions from a predefined pool. This pool can be curated from various sources, such as query logs or historical interactions.

4. **Re-ranking of Candidate Clarification Questions**: The system employs two fine-tuned BERT models to re-rank the candidate clarification questions. The first model, BERT-C-cq, learns an association between the conversation context and the clarification questions. The second model, BERT-C-P-cq, learns an association between the conversation context, the retrieved passages, and the clarification questions. The final ranking of the clarification questions is determined by combining the scores from both models.

5. **Selection of the Next Clarification Question**: The system selects the highest-ranked clarification question from the re-ranked list and presents it to the user.

The invention is particularly useful in scenarios where users need to explore complex or ambiguous information spaces, such as open-domain search or technical customer support. By leveraging deep learning models and passage retrieval, the system can generate more accurate and contextually relevant clarification questions, thereby improving the overall effectiveness of the conversational search process.

## DETAILED DESCRIPTION

### Experimental Results

To validate the effectiveness of the proposed method, extensive experiments were conducted on two diverse datasets: an open-domain search dataset (ClariQ) and an internal customer support dataset (Support). The ClariQ dataset consists of high-quality conversations with three turns: an initial user query, an agent clarification question, and a user response. The Support dataset contains noisy logs of human-to-human conversations, which required additional preprocessing to identify relevant clarification questions.

#### Experiment Setup

The experiments were conducted using the following setup:

- **Document Indexing**: The documents were indexed using Apache Lucene with an English language analyzer and default BM25 similarity. For the ClariQ dataset, the text field was used for retrieval, while for the Support dataset, the anchor and text fields were used.
- **Passage Retrieval**: Top-k documents were retrieved based on the conversation context, and candidate passages were extracted using a sliding window of 512 characters. Each passage was scored based on its coverage of terms in the conversation context, using a combination of global idf and scaled tf.
- **Clarification Question Retrieval**: The retrieved passages were used to query the Clarification-questions index, which contained a pool of candidate clarification questions.
- **Re-ranking with BERT Models**: Two BERT models, BERT-C-cq and BERT-C-P-cq, were fine-tuned to re-rank the candidate clarification questions. The first model learned an association between the conversation context and the clarification questions, while the second model learned an association between the conversation context, the retrieved passages, and the clarification questions. The final scores of the candidates were determined by combining the scores from both models using a simple Comb-SUM fusion.

#### Results

The results of the experiments are summarized in Table 2, which reports the Recall@30 metric on the development sets of the two datasets. On both datasets, each of the BERT re-rankers showed a significant improvement over the initial retrieval from the Clarification-questions index (IR-Base). For example, on the Support dataset, BERT-C-cq achieved a Recall@30 of 0.538, compared to 0.294 for IR-Base, representing an improvement of 82%.

When the scores from the two BERT models were fused, there was an additional improvement of about 2.5% over each of the rankers separately. For instance, on the ClariQ dataset, BERT-fusion achieved a Recall@30 of 0.791, compared to 0.77 for BERT-C-cq. This improvement can be attributed to the complementary matching that each of the two BERT models learns. The second model, BERT-C-P-cq, leverages latent features revealed through the retrieved passages, while the first model, BERT-C-cq, works better for cases where the retrieved passages are noisy.

The official ClariQ leaderboard results, shown in Table 3, further confirm the effectiveness of the proposed method. Our method, BERT-fusion, was ranked fourth but was the second-best as a team. It is worth noting that the top-performing system (NTES ALONG) gave preferences to clarification questions from the test data, which is not a valid assumption in general. In contrast, our method treats all clarification questions equally in the given pool.

### Conclusion

The invention provides a robust and effective method for selecting clarification questions in content-grounded conversational search scenarios. By leveraging deep learning models and passage retrieval, the system can generate more accurate and contextually relevant clarification questions, thereby improving the overall effectiveness of the conversational search process. The method has been validated through extensive experiments on two diverse datasets, demonstrating significant improvements over baseline retrieval methods. This invention has the potential to enhance user experience in a wide range of applications, from open-domain search to technical customer support.