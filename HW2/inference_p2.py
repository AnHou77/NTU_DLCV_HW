import train_p2 as p2
import os
import sys

if __name__ == '__main__':

    save_model_dir = './save_models'
    model_path = os.path.join(save_model_dir,'acgan_generatorl_499ep_0.8234.pth')
    
    save_image_path = sys.argv[1]
    p2.inference(model_path,save_image_path)