#!/bin/bash
wget -O 'vit_e4_0.9567.pth' 'https://www.dropbox.com/s/13kyws4aabcelbz/vit_e4_0.9567.pth?dl=1'

if [ "$SHELL" = "/bin/bash" ]; then
echo "--------------------------------"
echo "| your login shell is the bash |"
echo "--------------------------------"
else
echo "--------------------------------------------"
echo "| your login shell is not bash but $SHELL |"
echo "--------------------------------------------"
fi

images_path=$1
output_path=$2
python3 "inference_p1.py" $images_path $output_path