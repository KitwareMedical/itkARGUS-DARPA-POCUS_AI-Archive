import itk
from itk import TubeTK as tube

import numpy as np

import site
site.addsitedir("../ARGUS")

class ARGUS_onsd_roi_inference():
    
    def __init__(self):
        
        self.decision_distance = 5.0
        self.min_nerve_width = 1.5
        self.distance_from_nerve_top = 2.0
        
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
        
        num_methods = 2
        slices_of_interest = [[], []]
        for i in range(image_size[2]):
            # Method 0
            slices_of_interest[0].append((np.count_nonzero(ar_labels[i]==2),i))
            # Method 1
            obj_min, obj_max = self.get_object_min_max(ar_labels[i], 2, 0, self.min_nerve_width/image_spacing[0])
            if obj_min != None:
                obj_len = obj_max-obj_min
            else:
                obj_len = 0
            slices_of_interest[1].append((obj_len,i))
        
        width = np.zeros([num_methods])
        width_count = np.zeros([num_methods])
        slices_to_consider = 20
        min_l = np.zeros([num_methods]).astype(int)
        max_l = np.zeros([num_methods]).astype(int)
        for method in range(num_methods):
            slices_of_interest[method].sort(reverse = True)
            indxs = [x[1] for x in slices_of_interest[method][0:slices_to_consider]]
            summary_l = [x[1]//5 * 5 for x in slices_of_interest[method][0:slices_to_consider]]
            common_l = max(set(summary_l), key = summary_l.count)
            min_l[method] = common_l-1
            while min_l[method] in indxs:
                min_l[method] -= 1
            max_l[method] = common_l+1
            while max_l[method] in indxs:
                max_l[method] += 1
                
            for slice_num in range(min_l[method],max_l[method]):
                top=0
                while np.count_nonzero(ar_labels[slice_num,:,top] == 2) == 0 and top<image_size[1]-10:
                    top += 1
                measure_y = int(top + self.distance_from_nerve_top / image_spacing[1])
            
                if measure_y<image_size[1]-5:
                    for y in range(measure_y-2,measure_y+2):
                        class_2 = np.where(ar_labels[slice_num,:,y] == 2)
                        if len(class_2[0]) > 0:
                            min_x = class_2[0][0]
                            max_x = class_2[0][-1]
                            width[method] = max(width[method], (max_x-min_x) * image_spacing[0])
                            width_count[method] += 1
            
        classification = 0
        if np.mean(width) > self.decision_distance:
            classification = 1

        return classification, width, min_l, max_l