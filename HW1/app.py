import glob
import os
from sys import argv
import mean_iou_evaluate

masks = mean_iou_evaluate.read_masks('test')
print(masks)