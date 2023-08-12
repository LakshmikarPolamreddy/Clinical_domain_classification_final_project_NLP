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

## Folder and files details:
We show here two folders, power point presentation(using Google slides) and Our research paper. One folder is data that contains original data scraped from the MTsamples website as unbalanced_data.csv and also contains balanced_data.csv, created using NLP Augmenter. The folder scripts contain 12 scripts that we have generated for all the project activities from web scraping to deployment.

## Datasets:
We use the data from MTSamples.com website that offers access to a large
collection of transcribed digital medical reports and examples. The site is designed
to cater to both learning and working medical transcriptionists, providing them
with sample reports for various medical specialties and work types. These samples
are contributed by various transcriptionists and users and are intended for
reference purposes only. This website offers 5013 samples across 40 different
medical specialties

## Results:
We notice that BERT and XLNet models outperformed all
other models in terms of F1-score with 0.99 and BERT is computationally less
expensive when compared to XLNet. In this case also, KNN model takes just
0.29 sec for training and but its F1-score stands at 0.81. We run BERT model for just
20 epochs as the accuracy dramatically improves and loss converges from second
epoch.

## Conclusion:
In this project, we first scrape the medical transcription data with associated
clinical domains from the MTsamples.com website and then build several models
for clinical domain classification task. As we notice sub-optimal performance
of all the models due to imbalanced nature of the data, we performed data
augmentation using NLP augmenter and SMOTE algorithm. With this balanced
data, we again measured the performance of these models. We conclude that
BERT model outperformed all other models in terms of F1-score with 0.99 and
is computationally less expensive than XLNet. Our future work aims to focus on
gathering more real medical transcriptions data instead of generating synthetic
samples and then these models will be evaluated to identify the best model for
this task of clinical domain classification.
