from PIL import Image
import numpy as np
import cv2 as cv
import os
from pathlib import Path


Set5path = "./data/HR/set5"
Set5OutPath = "./data/LR/set5"
def GetSet5():

   files = list(Path(Set5path).glob("*.png"))
   print(len(files))
   files = sorted(files)
   
   if not os.path.exists(Set5OutPath+"/x2"):
       os.makedirs(Set5OutPath+"/x2",exist_ok=True)

   if not os.path.exists(Set5OutPath+"/x3"):
      os.makedirs(Set5OutPath+"/x3",exist_ok=True)

   if not os.path.exists(Set5OutPath+"/x4"):
      os.makedirs(Set5OutPath+"/x4",exist_ok=True)


   for f in files:
     img = cv.imread(f)
     file_name = os.path.basename(f)

     if img is None:
         print(f"Warning: Could not read image: {f}")
         continue

     smaller_img_x2 = cv.resize(img,None,fx = 0.5,fy = 0.5,interpolation=cv.INTER_CUBIC)
     if(not cv.imwrite(Set5OutPath+"/x2"+"/"+file_name,smaller_img_x2)):
        print("Couldnt write to the files")

     smaller_img_x3 = cv.resize(img,None,fx = 0.334,fy = 0.334,interpolation=cv.INTER_CUBIC)
     if(not cv.imwrite(Set5OutPath+"/x3"+"/"+file_name,smaller_img_x3)):
        print("Couldnt write to the files")

     smaller_img_x4 = cv.resize(img,None,fx =0.25,fy = 0.25,interpolation=cv.INTER_CUBIC)
     if(not cv.imwrite(Set5OutPath+"/x4"+"/"+file_name,smaller_img_x4)):
        print("Couldnt write to the files")


Set14path = "./data/HR/set14"
Set14Outpath ="./data/LR/set14"
def GetSet14():

   files = list(Path(Set14path).glob("*.png"))
   print(len(files))
   files = sorted(files)
   
   if not os.path.exists(Set14Outpath+"/x2"):
       os.makedirs(Set14Outpath+"/x2",exist_ok=True)

   if not os.path.exists(Set14Outpath+"/x3"):
      os.makedirs(Set14Outpath+"/x3",exist_ok=True)

   if not os.path.exists(Set14Outpath+"/x4"):
      os.makedirs(Set14Outpath+"/x4",exist_ok=True)


   for f in files:
       img = cv.imread(f)
       file_name = os.path.basename(f)
       if img is None:
           print(f"Warning: Could not read image: {f}")
           continue
        
       smaller_img_x2 = cv.resize(img,None,fx = 0.5,fy = 0.5,interpolation=cv.INTER_CUBIC)
       if(not cv.imwrite(Set14Outpath+"/x2"+"/"+file_name,smaller_img_x2)):
          print("Couldnt write to the files")
       smaller_img_x3 = cv.resize(img,None,fx = 0.334,fy = 0.334,interpolation=cv.INTER_CUBIC)
       if(not cv.imwrite(Set14Outpath+"/x3"+"/"+file_name,smaller_img_x3)):
          print("Couldnt write to the files")
       smaller_img_x4 = cv.resize(img,None,fx =0.25,fy = 0.25,interpolation=cv.INTER_CUBIC)
       if(not cv.imwrite(Set14Outpath+"/x4"+"/"+file_name,smaller_img_x4)):
          print("Couldnt write to the files")


urban100path = "./data/HR/urban100"
urban100Outpath = "./data/LR/urban100"
def Geturban100():

   files = list(Path(urban100path).glob("*.png"))
   print(len(files))
   files = sorted(files)
   
   if not os.path.exists(urban100Outpath+"/x2"):
       os.makedirs(urban100Outpath+"/x2",exist_ok=True)

   if not os.path.exists(urban100Outpath+"/x3"):
      os.makedirs(urban100Outpath+"/x3",exist_ok=True)

   if not os.path.exists(urban100Outpath+"/x4"):
      os.makedirs(urban100Outpath+"/x4",exist_ok=True)


   for f in files:
       img = cv.imread(f)
       file_name = os.path.basename(f)
       if img is None:
           print(f"Warning: Could not read image: {f}")
           continue
        
       smaller_img_x2 = cv.resize(img,None,fx = 0.5,fy = 0.5,interpolation=cv.INTER_CUBIC)
       if(not cv.imwrite(urban100Outpath+"/x2"+"/"+file_name,smaller_img_x2)):
          print("Couldnt write to the files")
       smaller_img_x3 = cv.resize(img,None,fx = 0.334,fy = 0.334,interpolation=cv.INTER_CUBIC)
       if(not cv.imwrite(urban100Outpath+"/x3"+"/"+file_name,smaller_img_x3)):
          print("Couldnt write to the files")
       smaller_img_x4 = cv.resize(img,None,fx =0.25,fy = 0.25,interpolation=cv.INTER_CUBIC)
       if(not cv.imwrite(urban100Outpath+"/x4"+"/"+file_name,smaller_img_x4)):
          print("Couldnt write to the files")


#to Get the downscaled images remove the comment of the following func one by one 
#GetSet5()
#GetSet14()
Geturban100()

print("Downscaling Completed")


