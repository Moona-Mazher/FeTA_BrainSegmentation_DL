# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 15:43:28 2021

@author: Abdul Qayyum
"""

#%% Feta efficient dataset module
import os
import numpy as np
import os
import nibabel as nib
from skimage import io
import cv2
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
from PIL import Image 
import PIL 
import imageio
import scipy.misc
import pandas as pd
import os
import glob
import SimpleITK as sitk
import ast
import json
import os
import nibabel as nib
from skimage import io
import cv2
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
from PIL import Image 
import PIL 
import imageio
import scipy.misc
import pandas as pd

path = '/raid/Home/Users/aqayyum/EZProj/FeTA2021/Training/'
save_img="/raid/Home/Users/aqayyum/EZProj/FeTA2021/trainingdata/images/"
save_msk="/raid/Home/Users/aqayyum/EZProj/FeTA2021/trainingdata/masks/"
patients = os.listdir(f'{path}')
len(patients)
pathimg1=[]
data = {
    'patient': [],
    'patientid': [],
    'channel': []
}
for i in patients:
    print(i)
    pathim=os.path.join(path,i)
    pathimg=glob.glob(os.path.join(pathim, 'anat', '*_T2w.nii.gz'))[0]
    pathtestmask=glob.glob(os.path.join(pathim, 'anat', '*_dseg.nii.gz'))[0]
    sub = os.path.split(pathimg)[1].split('_')[0] # 
    pathimg1.append(pathimg)
    print(pathimg)
    # image data file
    img_data = nib.load(pathimg).get_fdata()
    # image data file
    # mask data file
    msk_data=nib.load(pathtestmask).get_fdata()
    
    for file in range(0,img_data.shape[2]):
        img=img_data[:,:,file]
        img = exposure.rescale_intensity(img, out_range='float')
        img = img_as_uint(img)
        msk=msk_data[:,:,file]
        msk=msk.astype(np.uint8)
        io.imsave(os.path.join(save_img,str(i.split("\\")[0])+"_"+str(file)+".png"),img)
        io.imsave(os.path.join(save_msk,str(i.split("\\")[0])+"_"+str(file)+".png"),msk)
#%% validation dataset
import os
import numpy as np
import os
import nibabel as nib
from skimage import io
import cv2
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
from PIL import Image 
import PIL 
import imageio
import scipy.misc
import pandas as pd
import os
import glob
import SimpleITK as sitk
import ast
import json
import os
import nibabel as nib
from skimage import io
import cv2
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
from PIL import Image 
import PIL 
import imageio
import scipy.misc
import pandas as pd

path1 = '/raid/Home/Users/aqayyum/EZProj/FeTA2021/Validation/'
save_img="/raid/Home/Users/aqayyum/EZProj/FeTA2021/validationdata/images/"
save_msk="/raid/Home/Users/aqayyum/EZProj/FeTA2021/validationdata/masks/"
patients = os.listdir(f'{path1}')
len(patients)
pathimg1=[]
data = {
    'patient': [],
    'patientid': [],
    'channel': []
}
for i in patients:
    print(i)
    pathim=os.path.join(path,i)
    pathimg=glob.glob(os.path.join(pathim, 'anat', '*_T2w.nii.gz'))[0]
    pathtestmask=glob.glob(os.path.join(pathim, 'anat', '*_dseg.nii.gz'))[0]
    sub = os.path.split(pathimg)[1].split('_')[0] # 
    pathimg1.append(pathimg)
    print(pathimg)
    # image data file
    img_data = nib.load(pathimg).get_fdata()
    # image data file
    # mask data file
    msk_data=nib.load(pathtestmask).get_fdata()
    
    for file in range(0,img_data.shape[2]):
        img=img_data[:,:,file]
        img = exposure.rescale_intensity(img, out_range='float')
        img = img_as_uint(img)
        msk=msk_data[:,:,file]
        msk=msk.astype(np.uint8)
        io.imsave(os.path.join(save_img,str(i.split("\\")[0])+"_"+str(file)+".png"),img)
        io.imsave(os.path.join(save_msk,str(i.split("\\")[0])+"_"+str(file)+".png"),msk)
        
print("done")