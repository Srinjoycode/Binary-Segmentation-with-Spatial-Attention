"""
Author: @ayushmankumar7

Paste this file (prepare_train_mask.py) in "leftImg8bit_trainvaltest/leftImg8bit". 
There are 3 folder in this directory - test, train, val. 
Paste the file inside the folder containing the 3 folders.

In Command Prompt or Terminal : 

            python prepare_train_image.py


Link for the label: 
            https://www.cityscapes-dataset.com/file-handling/?packageID=1

"""

import os 
import cv2 
import numpy as np 
import glob 

try:
    os.makedirs("train_image")
except:
    pass 

print("It might take a few time. Be patient! Let this program run in background.")


files = glob.glob("train/*")

for file in files:
    images = glob.glob(file+"\\*.png")
    for image in images:
        img_name = image.split("\\")[-1]
        img = cv2.imread(image)
        # print(f"train_image/{image}")
        cv2.imwrite(f"train_image/{img_name}", img)

        print(image, "----> DONE")
    

print("You train Images is stored in './train_image' successfully! \n You may proceed. \n Thank you")
