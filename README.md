# Clinical_domain_classification_final_project_NLP

## Motivation:
The exponential growth of clinical data from diverse sources such as electronic health records, medical literature, and patient-generated information necessitates advanced solutions for efficient medical record classification.
## Project Objective:
To accurately classify the medical specialties(clinical domains) based on  patient's medical transcriptions – to automatically predict the initial diagnostics needed for the patient and to refer the patient to the relevant department.
## Approach:
We attempt to explore the various models like traditional classifiers (
SVM, KNN, Decision-tree, Random forest, Naive Bayes, XGBoost), pre-trained
transformer models (BERT, XLNet) and Few-shot prompting. First, we test
the performance of all these models with the original dataset obtained from
MTsamples website. Later, we convert the original imbalanced dataset into
balanced dataset using NLP Augmenter and then measure the performance of
these models. In addition, we also apply SMOTE algorithm to create balanced
dataset and then performance is measured. Among all these models, we select
the best model in terms of F1 score and computational cost to deploy and then
to use for prediction of the clinical domain.

## Datasets:
We use the data from MTSamples.com website that offers access to a large
collection of transcribed digital medical reports and examples. The site is designed
to cater to both learning and working medical transcriptionists, providing them
with sample reports for various medical specialties and work types. These samples
are contributed by various transcriptionists and users and are intended for
reference purposes only. This website offers 5013 samples across 40 different
medical specialties
