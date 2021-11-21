import train_p3 as p3
import os
import sys

if __name__ == '__main__':
    
    target_data_path = sys.argv[1]
    target_domain_name = sys.argv[2]
    output_path = sys.argv[3]
    model_type = sys.argv[4]

    save_model_dir = './save_models'
    model_path = os.path.join(save_model_dir,f'dann_target_{target_domain_name}_{model_type}.pth')

    p3.inference(model_path,target_data_path,target_domain_name,output_path)