import train_p1 as p1
import os
import sys

if __name__ == '__main__':

    model_path = 'vit_e4_0.9567.pth'
    
    images_path = sys.argv[1]
    output_path = sys.argv[2]
    p1.inference(model_path,images_path,output_path)