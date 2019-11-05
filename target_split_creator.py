import numpy as np
import pickle

arr = np.array([[2,4], [1,5], [8,9], [11,12], [28,22], [26,34], [30,31],[56,69],[42,43],[13,18], [44,45], [30,31], [16,17], [60,61], [54,55], [65,66], [89,90], [72,80], [23,24], [52,53]], dtype=np.int64)
with open('sparse_simba/data/targeted_split.pickle', 'wb') as f:
  pickle.dump(arr,f)
