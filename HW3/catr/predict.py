import torch

from transformers import BertTokenizer
from PIL import Image
import argparse

from models import caption
from datasets import coco, utils
from configuration import Config
import os

import matplotlib.pyplot as plt
import cv2
import glob

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--input_path', type=str, help='path to the folder containing test images', required=True)
parser.add_argument('--output_path', type=str, help='path to the folder for your visualization outputs', required=True)
parser.add_argument('--v', type=str, help='version', default='v3')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
args = parser.parse_args()
image_path = args.input_path
output_path = args.output_path
version = args.v
checkpoint_path = args.checkpoint

config = Config()

if version == 'v1':
    model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
elif version == 'v2':
    model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
elif version == 'v3':
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
else:
    print("Checking for checkpoint.")
    if checkpoint_path is None:
      raise NotImplementedError('No model to chose from!')
    else:
      if not os.path.exists(checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
      print("Found checkpoint! Loading!")
      model,_ = caption.build_model(config)
      print("Loading Checkpoint...")
      checkpoint = torch.load(checkpoint_path, map_location='cpu')
      model.load_state_dict(checkpoint['model'])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

filenames = glob.glob(os.path.join(image_path, '*'))

for fn in filenames:
    image = Image.open(fn)
    image = coco.val_transform(image)
    image = image.unsqueeze(0)

    def create_caption_and_mask(start_token, max_length):
        caption_template = torch.zeros((1, max_length), dtype=torch.long)
        mask_template = torch.ones((1, max_length), dtype=torch.bool)

        caption_template[:, 0] = start_token
        mask_template[:, 0] = False

        return caption_template, mask_template


    caption, cap_mask = create_caption_and_mask(
        start_token, config.max_position_embeddings)


    @torch.no_grad()
    def evaluate():
        model.eval()
        for i in range(config.max_position_embeddings - 1):
            predictions, attn, img_size = model(image, caption, cap_mask)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)

            if predicted_id[0] == 102:
                return caption, attn, img_size

            caption[:, i+1] = predicted_id[0]
            cap_mask[:, i+1] = False

        return caption, attn, img_size

    output,attn,img_size = evaluate()
    image = Image.open(fn)
    image_name = fn.split('/')[-1]
    image_name = image_name.replace('jpg','png')
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    print(result.capitalize())
    vocab = result.split(' ')
    l = len(vocab) + 2
    nrow = l // 5 if l % 5 == 0 else l // 5 + 1
    ncol = 5

    cnt = 0
    fig, ax = plt.subplots(nrows=nrow,ncols=ncol,figsize=(48, 32))
    for r in range(nrow):
        for i in range(ncol):
            if r * ncol + (i+1) > l:
                ax[r,i].axis('off')
            else:
                if cnt == l-1:
                    titles = '<end>'
                elif cnt == 0:
                    titles = '<start>'
                else:
                    titles = vocab[cnt-1]

                if cnt == 0:
                    ax[r,i].set_title(titles,fontdict={'fontsize':48})
                    ax[r,i].imshow(image)
                    ax[r,i].axis('off')
                else:
                    att = attn[cnt-1].reshape(img_size).detach().numpy()
                    att = cv2.resize(att / att.max(), image.size)
                    ax[r,i].set_title(titles,fontdict={'fontsize':48})
                    ax[r,i].imshow(image)
                    ax[r,i].imshow(att,interpolation='gaussian',alpha=0.4,cmap='jet')
                    ax[r,i].axis('off')
                cnt += 1
    plt.savefig(os.path.join(output_path, image_name))