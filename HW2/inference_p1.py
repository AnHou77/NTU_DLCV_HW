import train_p1 as p1
import os
import sys

if __name__ == '__main__':

    save_model_dir = './models'
    model_path = os.path.join(save_model_dir,'dcgan_generator_584ep.pth')
    
    save_image_path = sys.argv[1]
    p1.inference(model_path,save_image_path)