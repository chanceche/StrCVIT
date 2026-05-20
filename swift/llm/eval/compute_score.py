from swift.llm.eval.cider import Cider
from swift.llm.eval.ciderD import CiderD

# 初始化 CIDEr 计算器
cider = Cider()

# 参考句子
gts = {'1': ['A group of people stand in the back of a truck filled with cotton.'], 
    '2': ['Men are standing on and about a truck carrying a white substance.'],
    '3': ['A group of people are standing on a pile of wool in a truck.'], 
    '4': ['A group of men are loading cotton onto a truck'],
     '5': ['Workers load sheared wool onto a truck.']}

# 候选句子，转换为字典格式，以 image_id 作为键
res = {
'1': ['A group of people are working in a field with cotton.'], 
'2': ['A group of people are working in a field with cotton.'], 
'3': ['A group of people are working in a field with cotton.'], 
'4': ['A group of people are working in a field with cotton.'], 
'5': ['A group of people are working in a field with cotton.']
}

# 计算 CIDEr 分数
score, scores = cider.compute_score(gts, res)

# 打印 CIDEr 分数，保留较高的精度
print(f"CIDEr Score: {score:.10f}")
print("Individual Scores: ")
for idx, s in enumerate(scores):
    print(f"Image {idx+1} CIDEr Score: {s:.10f}")

