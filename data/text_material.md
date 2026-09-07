# Customer review sentiment classification (synthetic business material)

Task: text_classification. Domain: customer_reviews. The text column contains customer comments; label is the supervised sentiment target. Output prediction is a class label. Evaluate with weighted F1 and accuracy; probability is optional and means positive-class probability for binary labels.

TF-IDF Logistic Regression uses TF-IDF text features followed by Logistic Regression. It requires scikit-learn and is suitable for short text classification when labeled examples are available. Fill missing text with an empty string. Unseen words are ignored by the fitted vocabulary. Keep vectorizer fitting inside the training split to avoid leakage. Unigram/bigram vocabulary and regularization C are tunable. This lightweight baseline does not understand long-range context or sarcasm.
