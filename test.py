import numpy as np

data = np.load('data/data_DD.npz', allow_pickle=True)
lst = data.files

for item in lst:
    #print(item)
    #print(data[item])

X_train = np.load('data/train_X.npy', mmap_mode='r')
print(X_train)
