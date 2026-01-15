This method tackles clarification question selection in conversational search ending with documents as answers. By leveraging passages combined with deep learning models, the quality of selected questions significantly improves. The approach was tested on two diverse datasets, ClariQ and Support, yielding improvements of 12% to 87% over base IR retrieval.

For the experiments, Apache Lucene was used for document indexing, configured with an English language analyzer and BM25 similarity. For the customer support dataset (Support), both anchor and text fields were utilized for initial document retrieval due to the presence of training conversations in most documents.

In contrast, for the open-domain ClariQ dataset, only the text field was used for retrieval despite its large number of documents. This approach avoids bias towards a small subset with anchor texts. A sliding window of 512 characters was applied on retrieved document content for passage extraction.

The full conversations were employed to retrieve passages, but when feeding data to BERT models, only the last m utterances fitting within 512 characters were used. The pytorch huggingface implementation of BERT was utilized, with fine-tuning using default hyperparameters and a maximum sequence length of 256 tokens for the BERT-C-cq model and 384 tokens for the BERT-C-P-cq model.

Results on both datasets showed significant improvements over initial retrieval. On Support, BERT-C-cq achieved R@30=0.538 compared to IR-Base's R@30=0.294, an improvement of 82%. Fusing the scores from both BERT models provided an additional 2.5% gain in performance on both datasets.

The official ClariQ leaderboard results placed our method fourth overall but second among teams. Despite this, it was noted that the top-performing system gave preference to test data clarification questions, a non-generalizable approach. Our method treated all clarification questions equally from the provided pool, demonstrating robustness and generalizability across different conversational search scenarios.

In conclusion, the proposed method effectively improves the quality of selected clarification questions in conversational search by integrating passages with deep learning models. This approach shows significant improvements over traditional IR methods, validating its effectiveness on diverse datasets like ClariQ and Support.