#!/usr/bin/env python
# coding: utf-8

# In[37]:


import os
import glob

import cv2
import itk
import numpy as np


# In[38]:


def convert(fname, thresh):
    imlist = []

    cap = cv2.VideoCapture(fname)
    if cap.isOpened():

        image_type = itk.Image[itk.UC,2]
        diff_filter = itk.SimilarityIndexImageFilter[image_type,image_type].New()

        count = 0
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            img_slice = itk.image_from_array(np.array(gray))
            if count == 0:
                base_img_slice = img_slice
                imlist.append([img_slice, frame_index])
                count += 1
            else:
                diff_filter.SetInput1(base_img_slice)
                diff_filter.SetInput2(img_slice)
                diff_filter.Update()
                indx = diff_filter.GetSimilarityIndex()
                if indx < thresh:
                    base_img_slice = img_slice
                    imlist.append([img_slice, frame_index])
                    count += 1
            frame_index += 1
        cap.release()
 
    return imlist


# In[45]:


def create_sparse_image(fname, target_count):
    count = 0
    thresh = 1.0
    low_thresh = 1.0
    step = 0.1
    step_dir = -1
    while count != target_count and step > 1e-10 and thresh > 0.5:
        imlist = convert(fname,thresh)
        count = len(imlist)
        if step_dir == 1 and count > target_count:
            step_dir = -1
            step = step * 0.5
        elif step_dir == -1 and count < target_count:
                low_thresh = thresh
                step_dir = 1
                step = step * 0.5
        thresh += step_dir * step
    if count > 10:
        print('   ERROR: Did not attain target count =', target_count)
        imlist = convert(fname,low_thresh)
        count = len(imlist)
        print('   Creating image with', count, 'slices using threshold =', low_thresh)
    else:
        print('   Success!')
    dir_name = 'MHA'
    if os.path.isdir(dir_name):
        fileList = glob.glob(dir_name+'/'+fname+'-?????-f?????.mha')
        for file in fileList:
            os.remove(file)
    else:
        os.mkdir(dir_name)
    for i in range(len(imlist)):
        img_slice, frame_index = imlist[i]
        outfname = fname+'-%05d-f%05d.mha' % (i, frame_index)
        itk.imwrite(img_slice, dir_name+'/'+outfname)


# In[46]:


#create_sparse_image('Vir_liftl_H1N1_case2_clip2.mp4',10)


# In[47]:


for name in glob.glob('*.m[po][4v]'):
    print(name)
    create_sparse_image(name,10)


# In[ ]:




