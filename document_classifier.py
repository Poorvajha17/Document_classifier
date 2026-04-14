import os
import math
import random

def train_model(folder_path):
    freq = {}
    total_words = {}
    vocab = set()

    categories = os.listdir(folder_path)

    for c in categories:
        path = os.path.join(folder_path, c)

        if not os.path.isdir(path):
            continue

        freq[c] = {}
        total_words[c] = 0

        files = os.listdir(path)

        for file in files:
            if file.endswith(".txt"):
                f = open(os.path.join(path, file), 'r')
                text = f.read().lower()
                f.close()

                text = text.replace('.', '')
                text = text.replace(',', '')

                words = text.split()

                for w in words:
                    if w not in freq[c]:
                        freq[c][w] = 0

                    freq[c][w] += 1
                    total_words[c] += 1
                    vocab.add(w)

    return freq, total_words, list(vocab)

def predict(doc, freq, total_words, vocab):
    text = doc.lower()
    text = text.replace('.', '')
    text = text.replace(',', '')
    words = text.split()

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

def predict_sampling(doc, freq, total_words, vocab, samples=500):
    text = doc.lower()
    text = text.replace('.', '')
    text = text.replace(',', '')
    words = text.split()

    classes = list(freq.keys())
    weights = {}

    for c in classes:
        weights[c] = 0

    V = len(vocab)

    for _ in range(samples):
        total_all = sum(total_words.values())

        r = random.random()
        cumulative = 0

        for c in classes:
            p = total_words[c] / total_all
            cumulative += p
            if r <= cumulative:
                chosen = c
                break

        weight = 1

        for w in words:
            count = freq[chosen].get(w, 0)
            prob = (count + 1) / (total_words[chosen] + V)
            weight *= prob

        weights[chosen] += weight

    total = sum(weights.values())

    for c in weights:
        weights[c] = weights[c] / total

    best_class = max(weights, key=weights.get)
    return best_class, weights

data_path = "data"

freq, total_words, vocab = train_model(data_path)

print("Training complete")
print("Vocabulary size:", len(vocab))

for c in total_words:
    print(c, ":", total_words[c], "words")

doc = "Patient is admitted in the hospital"

pred1, scores1 = predict(doc, freq, total_words, vocab)
pred2, scores2 = predict_sampling(doc, freq, total_words, vocab)

print("\nDocument:", doc)
print("Naive Bayes Prediction:", pred1)
print("Sampling Prediction:", pred2)

print("\nNaive Bayes Scores:")
print(scores1)

print("\nSampling Scores:")
print(scores2)