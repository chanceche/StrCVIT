from swift.llm.eval.cider import Cider
from swift.llm.eval.ciderD import CiderD

# Initialize CIDEr scorer
cider = Cider()

# Reference sentences
gts = {'1': ['A group of people stand in the back of a truck filled with cotton.'], 
    '2': ['Men are standing on and about a truck carrying a white substance.'],
    '3': ['A group of people are standing on a pile of wool in a truck.'], 
    '4': ['A group of men are loading cotton onto a truck'],
     '5': ['Workers load sheared wool onto a truck.']}

# Candidate sentences, as dict keyed by image_id
res = {
'1': ['A group of people are working in a field with cotton.'], 
'2': ['A group of people are working in a field with cotton.'], 
'3': ['A group of people are working in a field with cotton.'], 
'4': ['A group of people are working in a field with cotton.'], 
'5': ['A group of people are working in a field with cotton.']
}

# Compute CIDEr score
score, scores = cider.compute_score(gts, res)

# Print CIDEr score with higher precision
print(f"CIDEr Score: {score:.10f}")
print("Individual Scores: ")
for idx, s in enumerate(scores):
    print(f"Image {idx+1} CIDEr Score: {s:.10f}")

