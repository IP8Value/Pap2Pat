- The paper introduces a QA system designed to efficiently handle large documents while being robust against adversarial inputs. It achieves this by selecting only the minimal context needed to answer each question, rather than processing full documents.

- Experiments across 5 datasets show the approach can achieve up to 15x training and 13x inference speedups compared to existing methods, with comparable or better accuracy. The system outperformed state-of-the-art models on SQuAD-Adversarial by a large margin.

- The sentence selector module identifies a minimal set of relevant sentences for each question using an attention mechanism. It can be applied as a plug-and-play component to any QA model without requiring end-to-end training.

- Key insights from the work include that most questions in existing datasets can actually be answered based on small context snippets, and that selecting this minimal context improves robustness to adversarial inputs by filtering out irrelevant or misleading information.