import os
import math

def build_model(data_path):
    word_counts = {}
    category_totals = {}
    vocabulary = set()

    categories = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]

    for cat in categories:
        word_counts[cat] = {}
        category_totals[cat] = 0
        cat_folder = os.path.join(data_path, cat)

        for filename in os.listdir(cat_folder):
            if filename.endswith(".txt"):
                with open(os.path.join(cat_folder, filename), 'r') as f:
                    text = f.read().lower()
                    words = text.replace('.', '').replace(',', '').split()

                    for word in words:
                        word_counts[cat][word] = word_counts[cat].get(word, 0) + 1
                        category_totals[cat] += 1
                        vocabulary.add(word)

    return word_counts, category_totals, list(vocabulary)

path_to_data = "data"
counts, totals, vocab = build_model(path_to_data)

print("AI Training Complete!")
print(f"Total Unique Words (Vocabulary Size): {len(vocab)}")
for cat in totals:
    print(f"Category '{cat}' has {totals[cat]} total words.")

def classify_document(test_text, word_counts, category_totals, vocabulary):
    test_words = test_text.lower().replace('.', '').replace(',', '').split()
    
    scores = {}
    vocab_size = len(vocabulary)

    for cat in word_counts:
        log_prob = 0 
        
        for word in test_words:
            count = word_counts[cat].get(word, 0)
            
            word_probability = (count + 1) / (category_totals[cat] + vocab_size)
            
            log_prob += math.log(word_probability)
        
        scores[cat] = log_prob

    winner = max(scores, key=scores.get)
    return winner, scores
'''
new_doc = "The patient was seen by a doctor for treatment at the hospital"

prediction, all_scores = classify_document(new_doc, counts, totals, vocab)

print(f"\nNew Document: '{new_doc}'")
print(f"AI Prediction: {prediction}")
print("Raw Scores (Log-Scale):", all_scores)
'''
def classify_by_sampling(test_text, word_counts, category_totals, vocabulary, num_samples=1000):
    test_words = test_text.lower().replace('.', '').replace(',', '').split()
    results = {}
    vocab_size = len(vocabulary)

    for cat in word_counts:
        total_weight = 0
        
        for _ in range(num_samples):
            sample_weight = 1.0
            
            for word in test_words:
                count = word_counts[cat].get(word, 0)
                prob = (count + 1) / (category_totals[cat] + vocab_size)
                
                sample_weight *= prob
            
            total_weight += sample_weight
        
        results[cat] = total_weight / num_samples

    winner = max(results, key=results.get)
    return winner, results

test_doc = "The court heard testimony from the witness regarding the contract."

pred1, scores1 = classify_document(test_doc, counts, totals, vocab)

pred2, scores2 = classify_by_sampling(test_doc, counts, totals, vocab)

print(f"\nDocument to Classify: '{test_doc}'")
print(f"Standard Bayes Prediction: {pred1}")
print(f"Sampling-based Prediction: {pred2}")