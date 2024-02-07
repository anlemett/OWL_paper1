th = 0.6
all_scores = [0.1, 0.3, 0.7]
all_scores = [1 if score < th else 2 for score in all_scores]
print(all_scores)