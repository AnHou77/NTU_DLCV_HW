import pandas as pd

gt = pd.read_csv('./data/p1_data/val_gt.csv')
result = pd.read_csv('../../hw1-AnHou77/output.csv')


acc = 0
for i in range(len(gt)):
    gt_name = gt.iloc[i]['image_id']
    gt_label = gt.iloc[i]['label']

    gt_name = gt_name.replace("_","")

    result_label = result.loc[result['image_id'] == gt_name]['label'].item()

    if gt_label == result_label:
        acc += 1

print(acc/len(gt))