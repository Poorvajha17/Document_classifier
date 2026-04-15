import math
import random
import csv
from sklearn.model_selection import train_test_split


def load_csv(filepath):
    texts = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['Text'].strip())
            labels.append(row['Label'].strip())
    return texts, labels

def train_test_split(texts, labels, test_ratio=0.2, seed=42):
    random.seed(seed)
    data = list(zip(texts, labels))
    random.shuffle(data)
    split = int(len(data) * (1 - test_ratio))
    train = data[:split]
    test = data[split:]
    return train, test

def train_model(train_data):
    freq = {}      
    total_words = {} 
    vocab = set()

    for text, label in train_data:
        if label not in freq:
            freq[label] = {}
            total_words[label] = 0

        words = preprocess(text)

        for w in words:
            freq[label][w] = freq[label].get(w, 0) + 1
            total_words[label] += 1
            vocab.add(w)

    return freq, total_words, list(vocab)

def preprocess(text):
    for ch in ['.', ',', '!', '?', '"', "'", '(', ')', ':', ';', '-', '\n']:
        text = text.replace(ch, ' ')
    return text.lower().split()

def predict_naive_bayes(doc, freq, total_words, vocab):
    words = preprocess(doc)

    class_scores = {}
    V = len(vocab)
    total_all = sum(total_words.values())

    for c in freq:
        prior = math.log(total_words[c] / total_all)
        score = prior

        for w in words:
            count = freq[c].get(w, 0)
            prob = (count + 1) / (total_words[c] + V)
            score += math.log(prob)

        class_scores[c] = score

    best_class = max(class_scores, key=class_scores.get)
    return best_class, class_scores

def predict_likelihood_weighting(doc, freq, total_words, vocab, samples=1000):
    words = preprocess(doc)
    classes = list(freq.keys())
    V = len(vocab)
    total_all = sum(total_words.values())

    priors = {c: total_words[c] / total_all for c in classes}

    cumulative = []
    cumsum = 0.0
    for c in classes:
        cumsum += priors[c]
        cumulative.append((cumsum, c))

    sample_log_weights = []  

    for _ in range(samples):
        r = random.random()
        chosen = classes[-1] 
        for cum, c in cumulative:
            if r <= cum:
                chosen = c
                break

        log_weight = 0.0
        for word in words:
            count = freq[chosen].get(word, 0)
            prob = (count + 1) / (total_words[chosen] + V)
            log_weight += math.log(prob)

        sample_log_weights.append((chosen, log_weight))

    max_lw = max(lw for _, lw in sample_log_weights)

    weights = {c: 0.0 for c in classes}
    for chosen, log_weight in sample_log_weights:
        weights[chosen] += math.exp(log_weight - max_lw)

    total_weight = sum(weights.values())
    for c in weights:
        weights[c] /= total_weight

    best_class = max(weights, key=weights.get)
    return best_class, weights

def evaluate(test_data, freq, total_words, vocab):
    nb_correct = 0
    lw_correct = 0
    n = len(test_data)

    for text, true_label in test_data:
        nb_pred, _ = predict_naive_bayes(text, freq, total_words, vocab)
        lw_pred, _ = predict_likelihood_weighting(text, freq, total_words, vocab, samples=500)

        if nb_pred == true_label:
            nb_correct += 1
        if lw_pred == true_label:
            lw_correct += 1

    nb_acc = nb_correct / n * 100
    lw_acc = lw_correct / n * 100
    return nb_acc, lw_acc

if __name__ == "__main__":
    csv_path = "data/df_file.csv"
    texts, labels = load_csv(csv_path)

    print(f"Total documents : {len(texts)}")
    print(f"Classes : {sorted(set(labels))}")

    train_data, test_data = train_test_split(texts, labels, test_ratio=0.2)
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    freq, total_words, vocab = train_model(train_data)

    print(f"\nVocabulary size: {len(vocab)}")
    for c in sorted(total_words):
        print(f"  Class {c}: {total_words[c]} words")

    doc = "The stock market saw a sharp increase due to rising corporate profits and strong economic growth"
    #doc = "The team secured a convincing victory in the championship match with outstanding performance from the players"
    #doc = "The new artificial intelligence system improves data processing speed and enhances machine learning performance"
    #doc = "The parliament passed a new law after extensive debate among government officials and political leaders"
    #doc = "Doctors recommend regular exercise and a balanced diet to maintain good health and prevent diseases"
    #doc = "The government announced new tax policies affecting small businesses"

    print(f"\nDemo Document")
    print(f"Document: {doc}")

    nb_pred, nb_scores = predict_naive_bayes(doc, freq, total_words, vocab)
    print(f"\nNaive Bayes Prediction: {nb_pred}")
    print("Naive Bayes Scores :")
    for c in sorted(nb_scores):
        print(f"  Class {c}: {nb_scores[c]:.4f}")

    lw_pred, lw_weights = predict_likelihood_weighting(doc, freq, total_words, vocab, samples=1000)
    print(f"\nLikelihood Weighting Prediction: {lw_pred}")
    print("Likelihood Weights :")
    for c in sorted(lw_weights):
        print(f"  Class {c}: {lw_weights[c]:.4f}")

    nb_acc, lw_acc = evaluate(test_data, freq, total_words, vocab)
    print(f"\nNaive Bayes Accuracy: {nb_acc:.2f}%")
    print(f"Likelihood Weighting Accuracy: {lw_acc:.2f}%")
