import numpy as np
import cv2 as cv
import os
import json
from scipy.special import logsumexp
from tqdm import tqdm

import glob
import argparse

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


class Perspective:
    def __init__(self, img , FOV, THETA, PHI, interpolation = cv.INTER_AREA):
        self._img = img
        [self._height, self._width] = self._img.shape[0:2]
        self.wFOV = FOV
        self.THETA = THETA
        self.PHI = PHI
        self.hFOV = float(self._height) / self._width * FOV

        self.w_len = np.tan(np.radians(self.wFOV / 2.0))
        self.h_len = np.tan(np.radians(self.hFOV / 2.0))
        self.interpolation = interpolation


    def toEquirec(self,height,width):

        x,y = np.meshgrid(np.linspace(-180, 180,width),np.linspace(90,-90,height))

        x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))

        xyz = np.stack((x_map,y_map,z_map),axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv.Rodrigues(z_axis * np.radians(self.THETA))
        [R2, _] = cv.Rodrigues(np.dot(R1, y_axis) * np.radians(-self.PHI))

        R1 = np.linalg.inv(R1)
        R2 = np.linalg.inv(R2)

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R2, xyz)
        xyz = np.dot(R1, xyz).T

        xyz = xyz.reshape([height , width, 3])
        inverse_mask = np.where(xyz[:,:,0]>0,1,0)

        xyz[:,:] = xyz[:,:]/np.repeat(xyz[:,:,0][:, :, np.newaxis], 3, axis=2)


        lon_map = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),(xyz[:,:,1]+self.w_len)/2/self.w_len*self._width,0)
        lat_map = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),(-xyz[:,:,2]+self.h_len)/2/self.h_len*self._height,0)
        mask = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),1,0)

        persp = cv.remap(self._img, lon_map.astype(np.float32), lat_map.astype(np.float32), self.interpolation, borderMode=cv.BORDER_REFLECT)

        mask = mask * inverse_mask
        #mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        if len(persp.shape) == 3:
            persp = persp * mask[...,None]
        else:
            persp = persp * mask


        return persp , mask

def create_mask(height, width, inner_rect):
    # Create an empty mask
    mask = np.zeros((height, width))

    # Define the rectangle in the middle
    inner_top_left, inner_bottom_right = inner_rect

    # Define the outer rectangle (the full image)
    outer_top_left = (0, 0)
    outer_bottom_right = (width-1, height-1)

    # Draw filled rectangles on the mask. The rectangle in the middle has
    # intensity 1, and the outer rectangle has intensity 0. OpenCV subtracts
    # the inner rectangle from the outer one.
    cv.rectangle(mask, outer_top_left, outer_bottom_right, 0, thickness=-1)
    cv.rectangle(mask, inner_top_left, inner_bottom_right, 1, thickness=-1)

    # Smooth the mask with a large Gaussian kernel to create a fading effect
    mask = cv.GaussianBlur(mask, (99, 99), 0)

    return mask

def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out

def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out 

class_map = {
  "DontCare" : 0,
  "Car" :  1,
  "Pedestrian" :  2,
  "Truck":  3,
  "Cyclist" :  4,
  "Misc" :  5,
  "Van":  6,
  "Tram":  7,
  "Person_sitting":  8,
  "Bus":  9,
  "Motorcycle": 10,
  "Animal": 11,
}

color_map_inst = {
0 :[0, 0, 0],
1 :[245, 150, 100],
2 :[245, 230, 100],
3 :[150, 60, 30],
4 :[180, 30, 80],
5 :[255, 0, 0],
6 :[30, 30, 255],
7 :[200, 40, 255],
8 :[90, 30, 150],
9 :[125,125,125],
10 :[90, 30, 150],
11 :[245, 230, 100],}

color_map = {
  0 : [0, 0, 0],
  1 : [70, 70, 70], # building
  2: [100, 40, 40], # fence
  3: [55, 90, 80], # other
  4: [220, 20, 60], # pedestrian
  5: [153, 153, 153], # pole
  6: [157, 234, 50], # road line
  7: [128, 64, 128], # road
  8: [244, 35, 232], # side walk
  9: [107, 142, 35], # vegetation
  10: [0, 0, 142], # Vehicle
  11: [102, 102, 156], # Wall
  12: [220, 220, 0], # traffic sign
  13: [70, 130, 180], # sky
  14: [81, 0, 81], # ground
  15: [150, 100, 100], # bridge
  16: [230, 150, 140], # rail track
  17: [180, 165, 180], # guard rail
  18: [250, 170, 30], # traffic light
  19: [110, 190, 160], # static
  20: [170, 120, 50], # dynamic
  21: [45, 60, 150], # water
  22: [145, 170, 100], # terrein
  23: [150, 240, 255],
  24: [0, 0, 150],
  25: [255, 255, 50],
  26: [245, 150, 100],
  27: [255, 0, 0],
  28: [200, 40, 255],
  29: [30, 125, 255],
  30: [90, 30, 150],
  31: [250, 80, 100],
  32: [180, 30, 80],
  33: [255, 0, 0]
}

def depth_to_array(image):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the depth value of each pixel normalized between [0.0, 1.0].
    """
    array = image.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    return normalized_depth
 
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main(arg):
    args = arg
    input_path = args.data_path
    equi_save_path = os.path.join(input_path,"equirectangular")
    target_h = args.height
    for frame_id in tqdm(range(len(glob.glob(os.path.join(input_path,"labels/*.json"))))):
        
        for cam_num in [0,1,2,3,4,5,6]:

            rgb_img = cv.imread(os.path.join(input_path,"images_rgb/%04d_%d.png" %(frame_id, cam_num)))#, cv.IMREAD_UNCHANGED)
            semantic = cv.imread(os.path.join(input_path,"images_ss/%04d_%d.png" %(frame_id, 10+cam_num)))#, cv.IMREAD_UNCHANGED)
            instance = cv.imread(os.path.join(input_path,"images_is/%04d_%d.png" %(frame_id, 20+cam_num)))#, cv.IMREAD_UNCHANGED)
            depth = cv.imread(os.path.join(input_path,"images_depth/%04d_%d.png" %(frame_id, 30+cam_num)), cv.IMREAD_UNCHANGED)
            
            with open(os.path.join(input_path,"calib_rgb/%04d_%d.json" %(frame_id, cam_num)), 'r') as fp:
                calib = json.load(fp)
            #print(calib["rotation"])
            calib_num = cam_num
            rows, cols, _ = rgb_img.shape
            fov = np.deg2rad(calib["fov"])
            fy = rows/(2*np.tan(fov/2))
            fx = cols/(2*np.tan(fov/2))
            cx = rows//2
            cy = cols//2

            _fov = 100
            #create_mask(cols, rows, inner_rect)
            
            P = Perspective(rgb_img,calib["fov"],calib["rotation"][2],calib["rotation"][0])
            equ , mask = P.toEquirec(target_h,2*target_h)
            
            P = Perspective(np.zeros((rgb_img.shape[0],rgb_img.shape[1],rgb_img.shape[2])),_fov,calib["rotation"][2],calib["rotation"][0])
            _ , mask = P.toEquirec(target_h,2*target_h)
            
            # Smooth the mask with a large Gaussian kernel to create a fading effect
            mask = cv.GaussianBlur(np.float32(mask), (99,99), 0)
            mask = np.repeat(mask[:, :, np.newaxis], equ.shape[-1], axis=2)

            normalized = depth_to_array(depth)
            in_meters = 1000 * normalized


            u, v = np.meshgrid(range(cols),range(rows))
            x = (u - cx) * in_meters[v, u] / fx
            y = (v - cy) * in_meters[v, u] / fy
            z = in_meters[v, u]
            
            range_img  = np.linalg.norm(np.stack([x,y,z], axis=-1),axis=-1)
            
            # Panorama stitching
            # For RGB and Range we use an alpha bleinding
            P = Perspective(instance,calib["fov"],calib["rotation"][2],calib["rotation"][0], interpolation=cv.INTER_NEAREST)
            equ_instance, mask_instance = P.toEquirec(target_h,2*target_h)
            
            P = Perspective(semantic,calib["fov"],calib["rotation"][2],calib["rotation"][0], interpolation=cv.INTER_NEAREST)
            equ_semantic, _ = P.toEquirec(target_h,2*target_h)
            mask_instance = np.repeat(mask_instance[:, :, np.newaxis], equ.shape[-1], axis=2)
            
            P = Perspective(range_img,calib["fov"],calib["rotation"][2],calib["rotation"][0])
            equ_range, _ = P.toEquirec(target_h,2*target_h)
            mask_range = np.repeat(mask_instance[:, :, np.newaxis], equ.shape[-1], axis=2)
            
            if cam_num == 0:
                equ_stitched = equ
                equ_range_stitched = equ_range
                equ_instance_stitched = equ_instance
                equ_semantic_stitched = equ_semantic
            else:
                equ_stitched = mask * equ + (1-mask)*equ_stitched
                equ_range_stitched = mask[...,0] * equ_range + (1-mask[...,0])*equ_range_stitched
                equ_instance_stitched = mask_instance*equ_instance + (1-mask_instance) * equ_instance_stitched
                equ_semantic_stitched = mask_instance*equ_semantic + (1-mask_instance) * equ_semantic_stitched
        
        # convert range image to cm and uint16 (this give some loss in precision).
        # For now we accept a cm accuracy 
        equ_range_stitched *= 100
        # clip gt range to 500 m
        equ_range_stitched = np.where(equ_range_stitched>50000, np.NaN, equ_range_stitched).astype(np.uint16)
        # clip preview range to 100 m
        equ_range_prev = np.where(equ_range_stitched>10000, np.NaN, equ_range_stitched)/10000
        equ_range_prev = cv.applyColorMap(np.uint8(255*equ_range_prev),cv.COLORMAP_MAGMA)
        
        
        os.makedirs(equi_save_path, exist_ok=True)

        cv.imwrite(os.path.join(equi_save_path,"rgb_%04d.png" %(frame_id)), equ_stitched)
        cv.imwrite(os.path.join(equi_save_path,"labels_%04d.png" %(frame_id)), equ_instance_stitched)
        cv.imwrite(os.path.join(equi_save_path,"semantic_%04d.png" %(frame_id)), equ_semantic_stitched)
        cv.imwrite(os.path.join(equi_save_path,"range_%04d.png" %(frame_id)), equ_range_stitched)
        cv.imwrite(os.path.join(equi_save_path,"prange_%04d.jpg" %(frame_id)), equ_range_prev)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-data_path',
        default="/workspace/data/CARLA_HIGH_RES_LIDAR/Ladybug5_TEST__",
        type=str,
        help='data path')
    argparser.add_argument(
        '-height',
        default=2048,
        type=int,
        help='height of the resulting panorama')



    args = argparser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')
