import pandas as pd
import numpy as np

data1 = pd.read_csv('./result.csv')
data2 = pd.read_csv('./data/p1_data/val_gt.csv')

l1 = np.array(data2['image_id'])
l2 = np.array(data2['label'])

acc = 0
for i in range(len(l1)):
    result = data1.loc[data1['image_id'] == l1[i]]
    result = result['label'].item()
    if l2[i] == result:
        acc += 1

print(f'acc: {acc/len(l1)}')