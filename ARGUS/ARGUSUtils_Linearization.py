import numpy as np

from scipy.stats import multivariate_normal
from scipy.stats import mode

import skimage
from skimage import measure

import sys

import math

def get_depth_and_zoom_C52(im):
    y = get_ruler_points_C52(im)

    avg = 0
    count = 0
    yCenters = []
    for j in range(y.size-1):
        avg += y[j]
        count += 1
        if(y[j+1]-y[j]>5):
            avg /= count
            yCenters.append(avg)
            avg = 0
            count = 0
    avg += y[y.size-1]
    count += 1
    avg /= count
    yCenters.append(avg)
    avg = 0
    for j in range(len(yCenters)-1):
        avg += yCenters[j+1]-yCenters[j]
    avg /= len(yCenters)-1    
    
    tic_num = len(yCenters)
    tic_min = yCenters[0]
    tic_max = yCenters[len(yCenters)-1]
    tic_diff = avg
    if(tic_num==17):
        tic_depth = 16
        tic_scale = tic_diff/54.4375
    elif(tic_num==13):
        tic_depth = 12
        tic_scale = tic_diff/55.5
    elif(tic_num==11):
        tic_depth = 5
        tic_scale = tic_diff/38.0
    else:
        print("ERROR: Unknown image depth!")
        print("   Num tics = ", tic_num, "   diff = ", tic_diff)
        
    return tic_depth,tic_scale #,tic_num,tic_min,tic_max,tic_diff


def get_top_and_bottom_curve_points_C52(im):
    """ Find points along the top (yt) and bottom (yb) curves of the ultrasound image """
    centerX = centerX0
    min_x = int(centerX-centerX*0.1)
    max_x = int(centerX+centerX*0.1)
    step_x = 10
    size_x = int((max_x - min_x)/step_x)
    xt = np.zeros(size_x, dtype=int)
    yt = np.zeros(size_x, dtype=int)
    for t in range(size_x):
        xt[t] = int(min_x + t * step_x)
        mid = np.mean(im[:,xt[t]-2:xt[t]+2],axis=1)
        nz = np.flatnonzero(mid)
        yt[t] = nz[0]
    min_x = int(centerX-centerX*0.25)
    max_x = int(centerX+centerX*0.25)
    step_x = 10
    size_x = int((max_x - min_x)/step_x)
    xb = np.zeros(size_x, dtype=int)
    yb = np.zeros(size_x, dtype=int)
    for t in range(size_x):
        xb[t] = int(min_x + t * step_x)
        mid = np.mean(im[:,xb[t]-2:xb[t]+2],axis=1)
        nz = np.flatnonzero(mid)
        yb[t] = nz[nz.size-1]
    return xt,yt,xb,yb


def calculate_radius(xc, yc, x, y):
    """ calculate the distance of each data points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)


def calculate_radius_C52(yc, x, y):
    """ calculate the distance of each data points from the center (xc, yc) """
    return np.sqrt((x-centerX0)**2 + (y-yc)**2)


def fit_circle(c, x, y):
    """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
    Ri = calculate_radius(*c, x, y)
    return Ri - Ri.mean()


def fit_circle_C52(c, x, y):
    """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
    Ri = calculate_radius_C52(*c, x, y)
    return Ri - Ri.mean()


def get_ruler_points_C52(im):
    """ Find points along ruler on left side of image """
    y_min = 80
    y_max = 1080
    min_x = 10
    max_x = 12
    mid = np.mean((im[y_min:y_max,min_x:max_x]/200).astype(int),axis=1)
    y = np.flatnonzero(mid)
    return y


def get_linear_map_C52(mask, xt, yt, rt, xb, yb, rb, ray_density = 0.5, blur = 0.5):
    try:
        assert(blur > 0)
    except:
        sys.exit("blur needs to be greater than zero")

    midline = xt 

    center_x = xt 
    center_y = yt

    center = np.array([center_x, center_y])

    inner_radius = rt
    outer_radius = (yb+rb)-yt

    radii = np.array([inner_radius, outer_radius])
    angle = (67.5/180)*math.pi
    
    left_angle = -angle/2
    right_angle = angle/2

    # determine the x and y sizes of the resampled image
    # from ray density. y size will be sector depth

    target_xsize = int(ray_density*(outer_radius)*angle + 0.5) # arc length (pixels) times ray density
    target_ysize = int(outer_radius - inner_radius + 0.5) # depth of US image

    # create mapping tensor

    mapping = np.zeros([target_ysize, target_xsize, 11])

    thetas = np.linspace(left_angle, left_angle+angle, target_xsize+2)
    rads = np.linspace(inner_radius, outer_radius, target_ysize+2)

    s0 = mask.shape[0]
    s1 = mask.shape[1]
    for i in range(target_xsize):
        print(i/target_xsize)
        for j in range(target_ysize): 
            
            theta = thetas[i + 1]
            rad = rads[j + 1]
            
            x = np.sin(theta)*rad + center_x
            y = np.cos(theta)*rad + center_y
            
            kernel_center_x = int(np.round(x))
            kernel_center_y = int(np.round(y))

            kernel_weights = np.zeros([3,3])

            G = multivariate_normal([x,y], np.eye(2)*blur)
            for m in range(3):
                for n in range(3):
                    i0 = kernel_center_x + m - 1
                    i1 = kernel_center_y + n - 1
                    if (i0<s0 and i1<s1 and mask[i0,i1]):
                        kernel_weights[m,n] = G.pdf([i0,i1])
                    else:
                        kernel_weights[m,n] = 0.0

            if (np.sum(kernel_weights) != 0):
                kernel_weights = kernel_weights / np.sum(kernel_weights)
            kernel_weights = kernel_weights.reshape(9)
            mapping[j,i] = np.concatenate(([kernel_center_x, kernel_center_y], kernel_weights))
        
    return mapping


def get_linear_map_corners(mask, ray_density = 0.5, blur = 0.5):
#     mask is boolean array for US image
#     ray_density is number of rays per pixel in outer curve in US image
#     blur os variance of 2D gaussian used for weights

    try:
        assert(blur > 0)
    except:
        sys.exit("blur needs to be greater than zero")

    midline = np.shape(mask)[1] // 2 

    # get top left, top right, bottom left, bottom right "true" points of mask
    left_indices = np.asarray(np.where(mask[:,0:midline] == 1))
    right_indices = np.asarray(np.where(mask[:,midline:] == 1))

    tl = left_indices[:,np.argmin(left_indices[0])]
    tr = right_indices[:,np.argmin(right_indices[0])] + np.array([0,midline])
    bl = left_indices[:,np.argmin(left_indices[1])]
    br = right_indices[:,np.argmax(right_indices[1])] + np.array([0,midline])
    
    # get "center" point of circles in US image
    
    sl = (bl[1] - tl[1])/(bl[0]-tl[0])
    sr = (br[1] - tr[1])/(br[0]-tr[0])

    center_x = -((-sl* tl[0] + tl[1] + sr *tr[0] - tr[1])/(sl - sr)) 
    center_y = -((-sl* sr *tl[0] + sr* tl[1] + sl *sr *tr[0] - sl* tr[1])/(sl - sr))

    center = np.array([center_x, center_y])

    inner_radius = (np.linalg.norm(center - tl)+ np.linalg.norm(center - tr))/2
    outer_radius = (np.linalg.norm(center - bl)+ np.linalg.norm(center - br))/2

    radii = np.array([inner_radius, outer_radius])

    # determine total angle of the sector in the image

    left = np.linalg.norm(center - bl)
    right = np.linalg.norm(center - br)
    across = np.linalg.norm(bl - br)
    total_angle = np.arccos((left**2 + right**2 - across**2) / (2*left*right))

    # calculate bottom angle (angle of left side wrt midline)

    bottom_angle = np.arcsin((bl[1]-midline)/left)
    top_angle = np.arcsin((br[1]-midline)/right)

    # determine the x and y sizes of the resampled image
    # from ray density. y size will be sector depth

    target_xsize = int(ray_density*(outer_radius)*total_angle + 0.5) # arc length (pixels) times ray density
    target_ysize = int(outer_radius - inner_radius + 0.5) # depth of US image

    # create mapping tensor

    mapping = np.zeros([target_ysize, target_xsize, 11])

    thetas = np.linspace(bottom_angle, bottom_angle+total_angle, target_xsize + 2)
    rads = np.linspace(inner_radius, outer_radius, target_ysize + 2)

    for i in range(target_xsize):
        for j in range(target_ysize): 
            
            theta = thetas[i + 1]
            rad = rads[j + 1]
            
            x = np.cos(theta)*rad + center_x
            y = np.sin(theta)*rad + center_y
            
            kernel_center_x = int(np.round(x))
            kernel_center_y = int(np.round(y))

            kernel_weights = np.zeros([3,3])

            G = multivariate_normal([x,y], np.eye(2)*blur)
            for m in range(3):
                for n in range(3):
                    i0 = kernel_center_x + m - 1
                    i1 = kernel_center_y + n - 1
                    if (mask[i0,i1]):
                        kernel_weights[m,n] = G.pdf([i0,i1])
                    else:
                        kernel_weights[m,n] = 0.0

            if (np.sum(kernel_weights) == 0):
#                 print(i,j, "ij")
#                 print(kernel_center_x, kernel_center_y, "kc")
#                 sys.exit("sum of kernel weights was 0")
                kernel_weights = kernel_weights.reshape(9)
                mapping[j,i] = np.concatenate(([kernel_center_x, kernel_center_y], kernel_weights))
                continue

            kernel_weights = kernel_weights / np.sum(kernel_weights)
            kernel_weights = kernel_weights.reshape(9)

            mapping[j,i] = np.concatenate(([kernel_center_x, kernel_center_y], kernel_weights))
        
    return mapping


def get_mask_from_image(image):
    """ assumes center pixel belongs to US image """
    mask = image > 0.006 # threshold
    labelled = skimage.measure.label(mask)
    center = np.asarray(np.shape(image)) // 2
    num = labelled[center[0],center[1]]
    USmask = labelled == num
    return USmask
