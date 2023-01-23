import itk
from itk import TubeTK as tube

import numpy as np

class ARGUS_onsd_roi_inference():
    
    def __init__(self):
        
        self.decision_distance = 5.0
        self.min_nerve_width = 1.5
        self.distance_from_nerve_top = 2.0
        self.max_nerve_length = 1.0
        self.slice_scan_interval = 5
        
    def get_object_min_max(self, labels, obj_id=2, axis=0, thresh=10):
        arr = np.sum(labels==obj_id,axis=axis)
        obj_indx = np.where(arr>thresh)
        if len(obj_indx)>0 and len(obj_indx[0])>0:
            return obj_indx[0][0],obj_indx[0][-1]
        else:
            return None,None
        
    def inference(self, ar_image, ar_labels):
        image_size = ar_image.GetLargestPossibleRegion().GetSize()
        image_spacing = ar_image.GetSpacing()
        min_x = int(image_size[0]*0.1)
        max_x = int(image_size[0]*0.9)
        min_y = int(image_size[1]*0.3)
        max_y = int(image_size[1]*0.95)
        
        num_methods = 2
        slices_of_interest = [[], []]
        for i in range(image_size[2]):
            # Method 0
            slices_of_interest[0].append((np.count_nonzero(
                ar_labels[i,min_x:max_x,min_y:max_y]==2),i))
            # Method 1
            obj_min, obj_max = self.get_object_min_max(
                ar_labels[i,min_x:max_x,min_y:max_y],
                2,
                0,
                self.min_nerve_width/image_spacing[0])
            if obj_min != None:
                obj_len = obj_max-obj_min
            else:
                obj_len = 0
            slices_of_interest[1].append((obj_len,i))
        
        slices_to_consider = 20
        min_l = np.zeros([num_methods]).astype(int)
        max_l = np.zeros([num_methods]).astype(int)
        min_nerve_size = (self.min_nerve_width / image_spacing[0])
        max_width = 0
        for method in range(num_methods):
            slices_of_interest[method].sort(reverse = True)
            indxs = [x[1] for x in slices_of_interest[method][0:slices_to_consider]]
            summary_l = [x[1]//self.slice_scan_interval * self.slice_scan_interval for x in slices_of_interest[method][0:slices_to_consider]]
            common_l = max(set(summary_l), key = summary_l.count)
            min_l[method] = common_l-1
            while min_l[method] in indxs:
                min_l[method] -= 1
            max_l[method] = common_l+1
            while max_l[method] in indxs:
                max_l[method] += 1
                
            for slice_num in range(min_l[method],max_l[method]):
                top=0
                while np.count_nonzero(ar_labels[slice_num,:,top] == 2) < min_nerve_size / 2 and top<image_size[1]-10:
                    top += 1
                bottom = top + 1
                while np.count_nonzero(ar_labels[slice_num,:,bottom] == 2) >= min_nerve_size / 2 and bottom<image_size[1]-10:
                    bottom += 1
                min_nerve_y = int(top + self.distance_from_nerve_top / image_spacing[1])
                max_nerve_y = min(bottom, int(min_nerve_y + self.max_nerve_length / image_spacing[1]))
            
                width = []
                for y in range(min_nerve_y,max_nerve_y):
                    class_2 = np.where(ar_labels[slice_num,:,y] == 2)
                    if len(class_2[0]) > min_nerve_size:
                        min_x = class_2[0][0]
                        max_x = class_2[0][-1]
                        width.append((max_x-min_x) * image_spacing[0])
                if len(width)>1:
                    width.sort(reverse=True)
                    slice_width = width[int(len(width)*0.02)]
                    max_width = max(max_width, slice_width)
                
        classification = 0
        if max_width > self.decision_distance:
            classification = 1

        return classification, max_width, min_l, max_l
