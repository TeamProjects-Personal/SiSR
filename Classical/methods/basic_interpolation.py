from PIL import Image
import cv2 as cv
from pathlib import Path
import os

resultPath = "./../results/basic"

NearestPath = resultPath+"/nearest"
BilinearPath = resultPath+"/bilinear"
BicubicPath = resultPath+"/bicubic"

if not os.path.exists(NearestPath):
    os.makedirs(NearestPath,exist_ok=True)

if not os.path.exists(BilinearPath):
    os.makedirs(BilinearPath,exist_ok=True)

if not os.path.exists(BicubicPath):
    os.makedirs(BilinearPath,exist_ok=True)


#region Set 5
Set5path = "./../data/LR/set5"
def UpScale_Set5(interpolation_method=cv.INTER_NEAREST,result_path=""):
    
        filesx2 = list(Path(Set5path + "/x2").glob("*.png"))
        filesx3 = list(Path(Set5path + "/x3").glob("*.png"))
        filesx4 = list(Path(Set5path + "/x4").glob("*.png"))

        x2_path = result_path + "/x2"
        x3_path = result_path + "/x3"
        x4_path = result_path + "/x4"

        if not os.path.exists(x2_path):
            os.makedirs(x2_path,exist_ok=True)

        if not os.path.exists(x3_path):
            os.makedirs(x3_path,exist_ok=True)

        if not os.path.exists(x4_path):
            os.makedirs(x4_path,exist_ok=True)


        for f in filesx2:
            img = cv.imread(f)
            file_name = os.path.basename(f)

            if img is None:
                print(f"Warning: Could not read image: {f}")
                continue

            upscaled_img_x2 = cv.resize(img,None,fx = 2.0,fy = 2.0,interpolation=interpolation_method)
            if(not cv.imwrite(x2_path+"/" + file_name,upscaled_img_x2)):
                print("Couldnt write the file")

        for f in filesx3:
            img = cv.imread(f)
            file_name = os.path.basename(f)

            if img is None:
                print(f"Warning: Could not read image: {f}")
                continue

            
            upscaled_img_x3 = cv.resize(img,None,fx = 3.0,fy = 3.0,interpolation=interpolation_method)
            if(not cv.imwrite(x3_path+"/" + file_name,upscaled_img_x3)):
                print("Couldnt write the file")

        for f in filesx4:
            img = cv.imread(f)
            file_name = os.path.basename(f)

            if img is None:
                print(f"Warning: Could not read image: {f}")
                continue

            upscaled_img_x4 = cv.resize(img,None,fx = 4.0,fy = 4.0,interpolation=interpolation_method)
            if(not cv.imwrite(x4_path+"/" + file_name,upscaled_img_x4)):
                print("Couldnt write the file")
    
#region  Nearest Neighbour 
UpScale_Set5(interpolation_method=cv.INTER_NEAREST,result_path=NearestPath)
#endregion

#region Bilinear
UpScale_Set5(interpolation_method=cv.INTER_LINEAR,result_path=BilinearPath)
#endregion

#region BiCubic
UpScale_Set5(interpolation_method=cv.INTER_CUBIC,result_path=BicubicPath)
#endregion

#endregion

#region Set 14
Set14path = "./../data/LR/set14"
def UpScale_Set14(interpolation_method=cv.INTER_NEAREST,result_path=""):
    
        filesx2 = list(Path(Set14path + "/x2").glob("*.png"))
        filesx3 = list(Path(Set14path + "/x3").glob("*.png"))
        filesx4 = list(Path(Set14path + "/x4").glob("*.png"))

        x2_path = result_path + "/x2"
        x3_path = result_path + "/x3"
        x4_path = result_path + "/x4"

        if not os.path.exists(x2_path):
            os.makedirs(x2_path,exist_ok=True)

        if not os.path.exists(x3_path):
            os.makedirs(x3_path,exist_ok=True)

        if not os.path.exists(x4_path):
            os.makedirs(x4_path,exist_ok=True)


        for f in filesx2:
            img = cv.imread(f)
            file_name = os.path.basename(f)

            if img is None:
                print(f"Warning: Could not read image: {f}")
                continue

            upscaled_img_x2 = cv.resize(img,None,fx = 2.0,fy = 2.0,interpolation=interpolation_method)
            if(not cv.imwrite(x2_path+"/" + file_name,upscaled_img_x2)):
                print("Couldnt write the file")

        for f in filesx3:
            img = cv.imread(f)
            file_name = os.path.basename(f)

            if img is None:
                print(f"Warning: Could not read image: {f}")
                continue

            
            upscaled_img_x3 = cv.resize(img,None,fx = 3.0,fy = 3.0,interpolation=interpolation_method)
            if(not cv.imwrite(x3_path+"/" + file_name,upscaled_img_x3)):
                print("Couldnt write the file")

        for f in filesx4:
            img = cv.imread(f)
            file_name = os.path.basename(f)

            if img is None:
                print(f"Warning: Could not read image: {f}")
                continue

            upscaled_img_x4 = cv.resize(img,None,fx = 4.0,fy = 4.0,interpolation=interpolation_method)
            if(not cv.imwrite(x4_path+"/" + file_name,upscaled_img_x4)):
                print("Couldnt write the file")

  
#region  Nearest Neighbour 
UpScale_Set14(interpolation_method=cv.INTER_NEAREST,result_path=NearestPath)
#endregion

#region Bilinear
UpScale_Set14(interpolation_method=cv.INTER_LINEAR,result_path=BilinearPath)
#endregion

#region BiCubic
UpScale_Set14(interpolation_method=cv.INTER_CUBIC,result_path=BicubicPath)
#endregion

#endregion

#region Set urban100
urban100path = "./../data/LR/urban100"
def UpScale_urban100(interpolation_method=cv.INTER_NEAREST,result_path=""):
    
        filesx2 = list(Path(urban100path + "/x2").glob("*.png"))
        filesx3 = list(Path(urban100path + "/x3").glob("*.png"))
        filesx4 = list(Path(urban100path + "/x4").glob("*.png"))

        x2_path = result_path + "/x2"
        x3_path = result_path + "/x3"
        x4_path = result_path + "/x4"

        if not os.path.exists(x2_path):
            os.makedirs(x2_path,exist_ok=True)

        if not os.path.exists(x3_path):
            os.makedirs(x3_path,exist_ok=True)

        if not os.path.exists(x4_path):
            os.makedirs(x4_path,exist_ok=True)


        for f in filesx2:
            img = cv.imread(f)
            file_name = os.path.basename(f)

            if img is None:
                print(f"Warning: Could not read image: {f}")
                continue

            upscaled_img_x2 = cv.resize(img,None,fx = 2.0,fy = 2.0,interpolation=interpolation_method)
            if(not cv.imwrite(x2_path+"/" + file_name,upscaled_img_x2)):
                print("Couldnt write the file")

        for f in filesx3:
            img = cv.imread(f)
            file_name = os.path.basename(f)

            if img is None:
                print(f"Warning: Could not read image: {f}")
                continue

            
            upscaled_img_x3 = cv.resize(img,None,fx = 3.0,fy = 3.0,interpolation=interpolation_method)
            if(not cv.imwrite(x3_path+"/" + file_name,upscaled_img_x3)):
                print("Couldnt write the file")

        for f in filesx4:
            img = cv.imread(f)
            file_name = os.path.basename(f)

            if img is None:
                print(f"Warning: Could not read image: {f}")
                continue

            upscaled_img_x4 = cv.resize(img,None,fx = 4.0,fy = 4.0,interpolation=interpolation_method)
            if(not cv.imwrite(x4_path+"/" + file_name,upscaled_img_x4)):
                print("Couldnt write the file")

  
#region  Nearest Neighbour 
UpScale_urban100(interpolation_method=cv.INTER_NEAREST,result_path=NearestPath)
#endregion

#region Bilinear
UpScale_urban100(interpolation_method=cv.INTER_LINEAR,result_path=BilinearPath)
#endregion

#region BiCubic
UpScale_urban100(interpolation_method=cv.INTER_CUBIC,result_path=BicubicPath)
#endregion

#endregion

print("Upscaling Completed")