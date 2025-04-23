# Spam Detection
The project concerns spam detection in Emails to determine whether the mail is spam or not. It includes data analysis, data preparation, text mining and create model by using different machine learning algorithms and BERT model(Bidirectional Encoder Representations from Transformers) in conjunction with cross-validation methodologies.

This project was inspired from the research paper [AI-Generated Text Detection and ClassificationBasedonBERT Deep Learning Algorithm](https://arxiv.org/pdf/2405.16422).

## Data
The data comes from the E-Mail classification NLP dataset on Kaggle. The link is [here](https://www.kaggle.com/datasets/datatattle/email-classification-nlp). 

## Methodology
The model follows the following process.

## Data Loading
The training (SMS_train.csv) and test (SMS_test.csv) datasets are loaded using pandas. The categorical variables "Spam" and "Non-Spam" are mapped to numerical values 1 and 0, respectively. Training and validation subsets are created using the TextDataset class, which will handle the tokenization and preparation of the input data.

### Model
BertTokenizer is initialized for encoding text data, and the computation device is set to GPU if available, otherwise CPU. Stratified K-Fold Cross-Validation is used to maintain the proportion of Spam:Non-Spam values across training and validation folds. 

The model used is BertForSequenceClassification with two output labels. The optimizer and learning rate scheduler are configured to facilitate effective training dynanmics. The model is pretrained including the forward-backward passes, gradient clipping and optimization steps. Early stopping is also implemented based on the F1 scores to prevent overfitting.

## Evaluation
After each epoch, the performance is evaluated on the validation set using the F1 score, where the best-performing model across all folds is identified and retained for the final model training.

## Final Model
The best model is retrained on the entire dataset and used to generate prediction on the test set. The predictions are mapped back to the original labels in the end (From 1 and 0 to "Spam and Non-Spam") and saved.