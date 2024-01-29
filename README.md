# question-answering-hf
This repository uses Huggingface and haystack library to perform question-answering on SubjQA Dataset.

### Frequent Question Types
Count of Frequent Question Types is dispayed below. Frequent Question Type comprised of words: "What", "How", "Is", "Does", "Do", "Was", "Where", "Why".
![questiontypes](https://github.com/Anshita1Saxena/question-answering-hf/blob/main/images/ques_count.png)

## Progress on the SQuAD 2.0 benchmark 
The following benchmark image is captured from Papers with Code.
![benchmark_squad_sota](https://github.com/Anshita1Saxena/question-answering-hf/blob/main/images/squad-sota.png)

## Span Classification
To extract the answers from raw text. The start and end tokens of an answer span act as the labels that a model needs to predict. Process is illustrated in the following figure:
![span_classification](https://github.com/Anshita1Saxena/question-answering-hf/blob/main/images/extract_answers.png)
For extractive QA, we can actually start with a fine-tuned model since the structure of the labels remains the same across datasets.
Baseline transformer models that are fine-tuned on SQuAD 2.0
![Baseline_models](https://github.com/Anshita1Saxena/question-answering-hf/blob/main/images/baseline_qa_models.PNG)

## Dealing with long passages
For Non QA tasks, truncation can be applied as a strategy, however, answers can be found at the end of the passage for the QA tasks, hence, trucation is not the preferred approach.
![passage_length](https://github.com/Anshita1Saxena/question-answering-hf/blob/main/images/max_length_512.png)
Here to select the appropriate answer, we use sliding window approach.
![sliding_window](https://github.com/Anshita1Saxena/question-answering-hf/blob/main/images/sliding_window.png)

## Retriver-Reader Architecture for modern QA systems
![retriever_reader](https://github.com/Anshita1Saxena/question-answering-hf/blob/main/images/haystack_qa_pipeline_architecture.png)
Document store
A document-oriented database that stores documents and metadata which are provided to the retriever at query time.
Pipeline
Combines all the components of a QA system to enable custom query flows, merging documents from multiple retrievers, and more.

## Model Specification
Retriever uses the ElasticSearch as a document-store, and Reader uses the `deepset/minilm-uncased-squad2` model based on deepset library. 

Why not :hugs: Transformers?

Deepset `FARMReader`
Based on deepset’s FARM framework for fine-tuning and deploying transformers. Compatible with models trained using nlpt_pin01 Transformers and can load models directly from the Hugging Face Hub.

`TransformersReader`
Based on the QA pipeline from :hugs: Transformers. Suitable for running inference only.

Although both readers handle a model’s weights in the same way, there are some differences in the way the predictions are converted to produce answers:

In :hugs: Transformers, the QA pipeline normalizes the start and end logits with a softmax in each passage. This means that it is only meaningful to compare answer scores between answers extracted from the same passage, where the probabilities sum to 1. For example, an answer score of 0.9 from one passage is not necessarily better than a score of 0.8 in another. In FARM, the logits are not normalized, so inter-passage answers can be compared more easily.

The TransformersReader sometimes predicts the same answer twice, but with different scores. This can happen in long contexts if the answer lies across two overlapping windows. In FARM, these duplicates are removed.

## Model Evaluation
Recall metric is used for Retriever's evaluation purpose. It reported 95% accuracy with top 3 extracted answers.

There are two types of retrievers: Sparse (used BM25/Best Match 25 instead of TF-IDF as it is perfect for document scoring), and Dense Passage Retriever. A well-known limitation of sparse retrievers like BM25 is that they can fail to capture the relevant documents if the user query contains terms that don’t match exactly those of the review. One promising alternative is to use dense embeddings to represent the question and document, and the current state of the art is an architecture known as Dense Passage Retrieval (DPR). The main idea behind DPR is to use two BERT models as encoders for the question and the passage. DPR’s bi-encoder architecture for computing the relevance of a document and query is illusted below:
![dpr](https://github.com/Anshita1Saxena/question-answering-hf/blob/main/images/dense_passage_retrieval.png)

Comparison between BM25 and DPR is illustrated below:
![comp_bm25_dpr](https://github.com/Anshita1Saxena/question-answering-hf/blob/main/images/comparison_bm25_dpr.png)

In extractive QA, there are two main metrics that are used for evaluating readers:

Exact Match (EM)
A binary metric that gives EM = 1 if the characters in the predicted and ground truth answers match exactly, and EM = 0 otherwise. If no answer is expected, the model gets EM = 0 if it predicts any text at all.

F1-score
Measures the harmonic mean of the precision and recall.

Performance of BM and F1 is demonstrated below:
![bm_f1](https://github.com/Anshita1Saxena/question-answering-hf/blob/main/images/fine_tune_squad.png)

## Domain Adaptation
This technique is used for boosting the performance by compiling all the data into single format. Visualization of the SQuAD JSON format is demonstrated in the below figure.
![domain_adaptation](https://github.com/Anshita1Saxena/question-answering-hf/blob/main/images/domain_adaptation.png)

Comparison of Fine-tune on SQUAD and Fine-tune on SQUAD+SUBJQA is demonstrated in the below figure:
![comp_da](https://github.com/Anshita1Saxena/question-answering-hf/blob/main/images/comparison_squad_squad%2Bsubj.png)

As we only have 1,295 training examples in SubjQA while SQuAD has over 100,000, so we might run into challenges with overfitting. We can see that fine-tuning the language model directly on SubjQA results in considerably worse performance than fine-tuning on SQuAD and SubjQA. Comparison of Fine-tune on SQUAD, SQUAD+SUBJQA and SUBJQA is demonstrated in the below figure:
![comp](https://github.com/Anshita1Saxena/question-answering-hf/blob/main/images/fine_tune_squad_squad%2Bsubj_subj.png)
When dealing with small datasets, it is best practice to use cross-validation when evaluating transformers as they can be prone to overfitting. 


