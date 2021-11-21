#!/bin/bash

if [ "$SHELL" = "/bin/bash" ]; then
echo "--------------------------------"
echo "| your login shell is the bash |"
echo "--------------------------------"
else
echo "--------------------------------------------"
echo "| your login shell is not bash but $SHELL |"
echo "--------------------------------------------"
fi

target_data_path = $1
target_domain_name = $2
output_path = $3
python3 "inference_p3.py" $target_data_path $target_domain_name $output_path "improved"