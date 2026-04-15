# README – Document Classifier

## Project Title

Document Classification using Naive Bayes and Likelihood Weighting

---

## Overview

This project implements a text classification system that categorizes documents into five classes: Politics, Business, Technology, Sports, and Entertainment. The system uses two probabilistic approaches: Naive Bayes (exact inference) and Likelihood Weighting (sampling-based approximate inference).

---

## Dataset

The input dataset is in CSV format with the following columns:

* Text: The document content
* Label: The category of the document

The dataset contains five labels corresponding to different domains.

---

## Features Used

* Word frequency per class
* Total word count per class
* Vocabulary of unique words
* Prior probabilities of classes

---

## Preprocessing Steps

* Removal of punctuation and special characters
* Conversion of text to lowercase
* Tokenization into words

---

## Algorithms Used

### 1. Naive Bayes Classifier

### 2. Likelihood Weighting

---

## Workflow

* Load dataset from CSV file
* Preprocess text data
* Split data into training and testing sets
* Train model using frequency-based learning
* Predict using Naive Bayes and Likelihood Weighting
* Evaluate accuracy on test data
* Compare results of both models

---

## Evaluation

Accuracy is computed as:

Accuracy = (Number of correct predictions / Total predictions) × 100

Both models are evaluated on the same test dataset for comparison.

---

## Results

The system outputs:

* Vocabulary size
* Word distribution per class
* Predictions for sample documents
* Class-wise scores or probabilities
* Final accuracy of both models

---

## Requirements

* Python 3.x
* Standard libraries: math, random, csv

---

## How to Run

* Place dataset CSV file in the project directory
* Update file path in the script
* Run the Python script:

```bash
python document_classifier.py
```

---

## Conclusion

This project demonstrates how probabilistic models can be applied to text classification. Naive Bayes provides fast and exact predictions, while Likelihood Weighting offers an approximate inference approach using sampling. The comparison highlights the trade-off between efficiency and approximation in probabilistic machine learning methods.
