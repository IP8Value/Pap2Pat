# DESCRIPTION

## BACKGROUND

Reading comprehension is a fundamental skill in education and professional settings, particularly in academic environments where the ability to understand and synthesize complex information is crucial. Traditional methods of assessing reading comprehension, such as paper-based tests, have limitations in terms of scalability, adaptability, and the ability to provide immediate feedback. The advent of internet-based computerized assessment has introduced numerous advantages, including support for innovative item types, measurement of complex knowledge and skills, automated scoring, and adaptive testing. However, these benefits come with the challenge of developing a large volume of high-quality assessment items, especially for complex tasks like reading comprehension.

Automatic Item Generation (AIG) has emerged as a promising solution to this challenge. AIG involves the use of computational models to generate assessment items, reducing the reliance on human experts and increasing the efficiency of item development. Traditional AIG methods, such as template-based and rule-based approaches, have been effective for simpler content areas like mathematics but fall short when it comes to more complex tasks like reading comprehension. The complexity of generating reading passages and associated questions, which require a deep understanding of language and context, has limited the applicability of AIG in this domain.

Recent advancements in natural language processing (NLP) and machine learning, particularly the development of transformer-based language models, have opened new possibilities for AIG. These models, such as GPT-3, can generate coherent and contextually appropriate text, making them suitable for creating reading passages and comprehension questions. The integration of these models into the AIG process can significantly enhance the quality and quantity of generated items, addressing the limitations of traditional methods and enabling the creation of more sophisticated and adaptive assessments.

## SUMMARY

The present invention relates to a method and system for generating and scoring reading comprehension passages and associated questions using transformer-based language models. The method involves the following steps:

1. **Passage Generation**: Using a transformer-based language model, a source passage is generated based on a set of instructions and examples that define the desired format, subject, and narrative style. The model is conditioned on these inputs to produce a coherent and contextually appropriate text.

2. **Question and Answer Generation**: Once the source passage is generated, the model is used to generate a suite of questions and potential answers. This includes main-idea questions, title questions, comprehension questions, and vocabulary-in-context questions. The model is provided with examples of questions and answers to condition the output and ensure the generated questions are relevant and answerable.

3. **Distractor Generation**: For selected-response questions, the model generates distractors (incorrect answers) by creating alternative texts that are stylistically and topically similar to the source passage but differ in content. These alternative texts are used to generate potential distractors, which are then evaluated and selected based on their similarity to the passage and the likelihood of being chosen by test takers.

4. **Human Review and Fairness Check**: All generated materials, including passages, questions, correct answers, and distractors, undergo a rigorous review process by subject-matter experts (SMEs) and fairness reviewers. The review ensures that the materials are appropriate, coherent, and free from bias or offensive content.

5. **Item Scoring and Psychometric Analysis**: The generated items are administered to test takers, and their responses are scored using automated scoring engines. Psychometric analysis is performed to evaluate the quality of the items, including item difficulty, discrimination, and local item dependence. Based on the analysis, items that do not meet the desired psychometric properties are refined or discarded.

The invention provides a comprehensive framework for AIG that leverages the capabilities of modern language models to create high-quality reading comprehension assessments. This framework addresses the limitations of traditional AIG methods and supports the development of more complex and adaptive assessments, enhancing the efficiency and effectiveness of educational and professional testing.

## DETAILED DESCRIPTION

### Passage Generation

The first step in the AIG process is the generation of the source passage. This is achieved using a transformer-based language model, such as GPT-3, which is capable of generating coherent and contextually appropriate text. The model is provided with a set of instructions and examples that define the desired format, subject, and narrative style of the passage. For example, the instructions might specify that the passage should be a short paragraph from a high school textbook on a particular topic.

The model is conditioned on these inputs to produce a source passage that meets the specified criteria. The conditioning process involves providing the model with a few examples of the desired format and content, which helps the model understand the desired style and structure of the text. The generated passage is then evaluated based on its coherence, relevance, and adherence to the specified format. If the passage meets the desired criteria, it is retained for further processing.

### Question and Answer Generation

Once the source passage is generated, the next step is to generate a suite of questions and potential answers. The types of questions generated include main-idea questions, title questions, comprehension questions, and vocabulary-in-context questions. Each type of question serves a specific purpose in assessing different aspects of reading comprehension.

#### Main-Idea Questions

Main-idea questions are designed to assess the test taker's ability to identify the central theme or main point of the passage. The model is provided with examples of passages and their associated main ideas to condition the output. The model generates multiple potential answers, which are then evaluated based on their similarity to the passage and the average negative log likelihood as estimated by the language model. The similarity is computed using vector representations of the passage and each candidate answer, and the negative log likelihood is derived from the model's output distribution. The best candidate answer is selected based on these metrics.

#### Title Questions

Title questions are designed to assess the test taker's ability to generate an appropriate title for the passage. The model is provided with examples of passages and their associated titles to condition the output. The process of generating and evaluating title questions is similar to that of main-idea questions, with the model generating multiple potential titles and selecting the best candidate based on similarity and negative log likelihood.

#### Comprehension Questions

Comprehension questions are designed to assess the test taker's understanding of specific details and concepts within the passage. The model is provided with examples of passages and their associated comprehension questions and answers to condition the output. The model generates new questions and their answers for the source passage. To ensure that the generated questions are answerable using the passage, an external question-answering model is used to predict the likelihood that the question can be answered. Questions with a low answerability likelihood, extremely long questions, and questions with very short answers are filtered out. The remaining questions are evaluated based on their similarity to the passage and the negative log likelihood of the answers.

#### Vocabulary-in-Context Questions

Vocabulary-in-context questions are designed to assess the test taker's understanding of the meaning of words in the context of the passage. The model is used to iteratively complete each word in the source passage, computing likelihoods for each word in its vocabulary. Candidate words for deletion are then filtered based on the likelihood and rank order of the original word being suggested by the model, syntactic information about the word, semantic information about the word, and the distance between the original word and nearby successful candidates. Distractors for deleted words are selected from the model's likelihood output for all other words in its vocabulary. Successful distractors have low, but not too low, likelihood and have the same syntactic part-of-speech as the correct answer.

### Distractor Generation

For selected-response questions, the generation of distractors (incorrect answers) is a critical step. The model generates alternative texts that are stylistically and topically similar to the source passage but differ in content. These alternative texts are used to generate potential distractors, which are then evaluated and selected based on their similarity to the passage and the likelihood of being chosen by test takers. The evaluation process involves computing a suite of NLP metrics, including the average similarity of the answer to other correct answer candidates, similarity of the answer to the source passage and to individual sentences in the source passage, and the model's estimated probability of generating the candidate answer. The best distractors are selected based on these metrics.

### Human Review and Fairness Check

All generated materials, including passages, questions, correct answers, and distractors, undergo a rigorous review process by subject-matter experts (SMEs) and fairness reviewers. The review process ensures that the materials are appropriate, coherent, and free from bias or offensive content. SMEs evaluate the cohesion, clarity, and logical consistency of the passages and the viability of each option in the questions. Fairness reviewers ensure that the materials do not contain any content that is too culturally specific, has technical or field-specific jargon, or could be potentially sensitive to test takers. Following the review, materials that require significant edits are discarded, and the remaining materials are refined and finalized.

### Item Scoring and Psychometric Analysis

The generated items are administered to test takers, and their responses are scored using automated scoring engines. For selected-response questions, the scoring is straightforward, with the correct answer receiving full credit and the distractors receiving no credit. For open-ended questions, such as text highlighting, a continuous grade between 0 and 1 is calculated based on the discrepancy between the response and the correct answer. The discrepancy is measured as the geometric distance between the start and end indexes of the selection and the correct answer.

Psychometric analysis is performed to evaluate the quality of the items, including item difficulty, discrimination, and local item dependence. Item difficulty is measured as the mean score on the item, and item discrimination is measured using item-total correlations. Local item dependence is investigated using partial correlations between pairs of items, controlling for the total practice test score. Items that do not meet the desired psychometric properties are refined or discarded.

### Future Directions

The invention provides a robust framework for AIG that can be extended to other types of texts and tasks. For example, argumentative texts of multiple viewpoints can be generated to support synthesis tasks, and different response formats, such as dragging and dropping, free response (both spoken and written), and interacting with other media, can be accommodated. The digital format of the assessment allows for the integration of multiple types of response formats and the delivery of comprehension questions in different modalities, such as aurally. These extensions can enhance the assessment's ability to approximate real-world reading scenarios and provide a more comprehensive evaluation of reading comprehension skills.

### Conclusion

The invention demonstrates how recent advances in computational language modeling can transform item development for complex tasks and assessments. By leveraging the capabilities of transformer-based language models, the invention provides a comprehensive framework for generating and scoring reading comprehension passages and associated questions. This framework addresses the limitations of traditional AIG methods and supports the development of more sophisticated and adaptive assessments, enhancing the efficiency and effectiveness of educational and professional testing.