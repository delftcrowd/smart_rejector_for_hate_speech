import pickle
import random

file = open("sem_eval_all.pkl", "rb")
data = pickle.load(file)
file.close()
hate = list(filter(lambda d: d['label'] == 'hate', data))
random_hate = random.choice(hate)
print(random_hate)
