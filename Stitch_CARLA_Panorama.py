import numpy as np
import open3d as o3d
import cv2 as cv
import os
import json
from scipy.special import logsumexp
from tqdm import tqdm
from utils.spherical import spherical_projection, get_equirectangular_coordinates
import copy
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def to_ego(pose, world2sensor):
    pose_cloud = o3d.geometry.PointCloud()
    pose_cloud.points = o3d.utility.Vector3dVector(pose.reshape(-1,3).astype(np.float64))
    pose_cloud.transform(world2sensor)
    
    ego_pose_trajectory = np.asarray(pose_cloud.points).reshape(pose.shape)
    return ego_pose_trajectory

BONE_KEYS = ["crl_hips__C", # Pelvis, 0
          "crl_thigh__L", # L_Hip, 1
          "crl_thigh__R", # R_Hip, 2
          "crl_spine__C", # Spine1, 3
          "crl_leg__L", # L_Knee, 4
          "crl_leg__R", # R_Knee, 5
          "crl_spine01__C", # Spine2, 6
          "crl_foot__L", # L_Ankle, 7
          "crl_foot__R", # R_Ankle, 8
          "crl_toe__L", # L_Foot, 9
          "crl_toe__R", # R_Foot, 10
          "crl_neck__C", # Neck, 11
          "crl_Head__C", # Head, 12
          "crl_shoulder__L", # L_Shoulder, 13
          "crl_shoulder__R", # R_Shoulder, 14
          "crl_arm__L", # L_Elbow, 15
          "crl_arm__R", # R_Elbow, 16
          "crl_foreArm__L", # L_Wrist, 17
          "crl_foreArm__R", # R_Wrist, 18
          "crl_hand__L", # L_Hand, 19
          "crl_hand__R", # R_Hand, 20
          ]

kinetic_tree = np.array([
    [0, 1], # Pelvis -> L_Hip
    [0, 2], # Pelvis -> R_Hip
    [0, 3], # Pelvis -> Spine1
    [1, 4], # L_Hip -> L_Knee
    [2, 5], # R_Hip -> R_Knee
    [3, 6], # Spine1 -> Spine2
    [4, 7], # L_Knee -> L_Ankle
    [5, 8], # R_Knee -> R_Ankle
    [7, 9], # L_Ankle -> L_Foot
    [8, 10], # R_Ankle -> R_Foot
    [6, 11], # Spine2 -> Neck
    [11, 12], # Neck -> Head
    [11, 13], # Neck -> L_Shoulder
    [11, 14], # Neck -> R_Shoulder
    [13, 15], # L_Shoulder -> L_Elbow
    [14, 16], # R_Shoulder -> R_Elbow
    [15, 17], # L_Elbow -> L_Wrist
    [16, 18], # R_Elbow -> R_Wrist
    [17, 19], # L_Wrist -> L_Hand
    [18, 20], # R_Wrist -> R_Hand
]).T


def o3d_draw_skeleton(joints3D, kintree_table, color_set="RGB"):

    colors = []
    if color_set == "CMY":
        left_right_mid = [[1.0,1.0,0.0], [0.0,1.0,1.0], [1.0,0.0,1.0]]
    else:
        left_right_mid = [[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]]
    kintree_colors = [0,1,2,0,1,2,0,1,2,1,2,0,0,1,2,1,2,1,2,1,2]
    for c in kintree_colors:
        colors.append(left_right_mid[c])
        
    
    # For each 20 joint
    lines = []
    points = []
    for i in range(1, kintree_table.shape[1]):
        j1 = kintree_table[0][i]
        j2 = kintree_table[1][i]
        lines.append([j1, j2])
    points = joints3D[:, :]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    joints = o3d.geometry.PointCloud()
    joints.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    joints.colors = o3d.utility.Vector3dVector(colors)
    return line_set, joints

def project_poses(label_file, imgrgb, pcd):
    calib_file = label_file.replace("labels","calib_rgb").replace(".json", "_0.json")
    calib = json.load(open(calib_file, 'r'))
    with open(label_file, 'r') as fp:
        data_raw = json.load(fp)
        #all_walker = {key: {bone_key: data_raw[key]["bones"][bone_key]["world"] for bone_key in BONE_KEYS} for key in data_raw if "walker" in data_raw[key]["type_id"]}
        line_sets = []
        joint_sets = []
        vrus = {}
        vehicles = {}
        W2S = np.array(calib["world2sensor"])
        W2S = np.linalg.inv(W2S)
        #pcd_world = pcd.transform(np.linalg.inv(W2S))
        #dynamic_instance_keys = np.unique(data['ObjIdx'])
        bboxes = []
        #pcd.transform(W2S)

        for key in data_raw.keys(): 
            if key == 0:
                continue
            key = str(key)

            center = data_raw[key]["location"]

            extent = data_raw[key]["extent"]

            l,h,w = data_raw[key]["extent"]
            x, y, z = data_raw[key]["location"]
            
            rotation_x, rotation_y, rotation_z = np.deg2rad(data_raw[key]["rotation"])
            
            direct = [x, y, z,rotation_x, rotation_y, rotation_z]
            
            center = np.array([[x,y,z]]).astype(np.float64).T

            extent = np.array([[2*l+0.5,2*h+0.5,2*w+0.5]]).astype(np.float64).T
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([rotation_x,rotation_z,rotation_y]).astype(np.float64)
            obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
            obb.rotate(W2S[:3, :3], center=(0, 0, 0))
            obb.translate(W2S[:3, 3])
            
            
            
            
            if np.asarray(pcd.crop(obb).points).shape[0] < 32:
                continue
            
            
            if not "walker" in data_raw[key]["type_id"]:
                if not key in vehicles:
                    vehicles[key] = data_raw[key]
                vehicles[key]["verts"] = to_ego(np.array(data_raw[key]["verts"]),W2S)
                #continue
            
            else:
                # get all bones in 3D coordinates
                bones_list = [data_raw[key]["bones"][bone_key]["world"] for bone_key in BONE_KEYS]  
                
                if not key in vrus:
                    vrus[key] = {"joints_world": [bones_list], "orientation": [direct]}
                else:
                    vrus[key]["joints_world"].append(bones_list)
                    vrus[key]["orientation"].append(direct)
                
                
                vrus[key]["joints_world"] = to_ego(np.array(vrus[key]["joints_world"]),W2S)
                

        return {"pedestrians": vrus, "vehicles": vehicles}

def rotate_point_cloud(xyz, theta, phi):
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, _] = cv.Rodrigues(z_axis * np.radians(theta))
    [R2, _] = cv.Rodrigues(np.dot(R1, y_axis) * np.radians(phi))

    R1 = np.linalg.inv(R1)
    R2 = np.linalg.inv(R2)

    xyz = xyz.reshape(-1, 3).T
    xyz = np.dot(R2, xyz)
    xyz = np.dot(R1, xyz).T
    return xyz

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

def min_max(pc, max_bound, min_bound):
    return (pc - min_bound)/(max_bound-min_bound)


def softmax(x, axis=None):
     return np.exp(x - logsumexp(x, axis=axis, keepdims=True))
 
def spherical_projection(pc, height=2048, width=2*2048, theta_range=[-np.pi/2,np.pi/2], th=0.01, sort_largest_first=False, bins_h=None, max_range=None):
    def to_deflection_coordinates(x,y,z):
        # To cylindrical
        p = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        # To spherical   
        theta = -np.arctan2(p, z) + np.pi/2
        return phi, theta
    
    '''spherical projection 
    Args:
        pc: point cloud, dim: N*C
    Returns:
        pj_img: projected spherical iamges, shape: h*w*C
    '''

    # filter all small range values to avoid overflows in theta min max calculation
    #if isinstance(theta_range, type(None)):
        
    r = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2)
    arr1inds = r.argsort()
    if sort_largest_first:
        pc = pc[arr1inds]
    else:
        pc = pc[arr1inds[::-1]]
    #pc = pc[arr1inds]
    r = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2)
    if not isinstance(max_range,type(None)):
        indices = np.where((r > th)*(r<=max_range))
    else:
        indices = np.where(r > th)
    pc = pc[indices]
        
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        
    phi, theta = to_deflection_coordinates(x,y,z)

    #indices = np.where(r > th)
    if isinstance(theta_range, type(None)):
        theta_min, theta_max = [theta.min(), theta.max()]
    else: 
        theta_min, theta_max = theta_range
        
    #phi_min, phi_max = [phi.min(), phi.max()]
    phi_min, phi_max = [-np.pi, np.pi]

    # assuming uniform distribution of rays
    if isinstance(bins_h, type(None)):
        bins_h = np.linspace(theta_min, theta_max, height)[::-1]
    bins_w = np.linspace(phi_min, phi_max, width)[::-1]

    theta_img = np.stack(width*[bins_h], axis=-1)
    phi_img = np.stack(height*[bins_w], axis=0)

    idx_h = np.digitize(theta, bins_h)-1
    idx_w = np.digitize(phi, bins_w)-1

    pj_img = np.zeros((height, width, pc.shape[1])).astype(np.float32)
    #pj_img_norm = np.zeros((height, width, pc.shape[1])).astype(np.float32)

    pj_img[idx_h, idx_w, :] += pc
    #pj_img_norm[idx_h, idx_w, :] += normilize_map

    #pj_img = pj_img/pj_img_norm

    alpha = np.sqrt(np.square(theta_img)+np.square(phi_img))

    return pj_img, alpha, (theta_min, theta_max), (phi_min, phi_max) 

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

for frame_id in tqdm(range(0,1000)):

    pcd_list = []
    pcd_masked_list = []
    pcd_masked_comb = o3d.geometry.PointCloud()
    pcd_comb = o3d.geometry.PointCloud()
    pcd_comb_labels = o3d.geometry.PointCloud()

    pano_list = []
    mask_list = []
    colors = []
    pcd_comb = o3d.geometry.PointCloud()
    with open("/workspace/data/CARLA_HIGH_RES_LIDAR/Ladybug5_0004/calib_rgb/%04d_%d.json" %(frame_id, 0), 'r') as fp:
        ref_calib = json.load(fp)
        
    label_file = "/workspace/data/CARLA_HIGH_RES_LIDAR/Ladybug5_0004/labels/%04d.json" %(frame_id)
    
    for cam_num in [0,1,2,3,4,5,6]:

        rgb_img = cv.imread("/workspace/data/CARLA_HIGH_RES_LIDAR/Ladybug5_0004/images_rgb/%04d_%d.png" %(frame_id, cam_num))#, cv.IMREAD_UNCHANGED)
        semantic = cv.imread("/workspace/data/CARLA_HIGH_RES_LIDAR/Ladybug5_0004/images_ss/%04d_%d.png" %(frame_id, 10+cam_num))#, cv.IMREAD_UNCHANGED)
        instance = cv.imread("/workspace/data/CARLA_HIGH_RES_LIDAR/Ladybug5_0004/images_is/%04d_%d.png" %(frame_id, 20+cam_num))#, cv.IMREAD_UNCHANGED)
        depth = cv.imread("/workspace/data/CARLA_HIGH_RES_LIDAR/Ladybug5_0004/images_depth/%04d_%d.png" %(frame_id, 30+cam_num), cv.IMREAD_UNCHANGED)
        
        with open("/workspace/data/CARLA_HIGH_RES_LIDAR/Ladybug5_0004/calib_rgb/%04d_%d.json" %(frame_id, cam_num), 'r') as fp:
            calib = json.load(fp)
        #print(calib["rotation"])
        calib_num = cam_num
        rows, cols, _ = rgb_img.shape
        fov = np.deg2rad(calib["fov"])
        fy = rows/(2*np.tan(fov/2)) #5.542562584220e+02
        fx = cols/(2*np.tan(fov/2))#5.542562584220e+02
        cx = rows//2#9.600000000000e+02
        cy = cols//2#3.600000000000e+02

        target_h = 2048
        #create_mask(cols, rows, inner_rect)
        
        P = Perspective(rgb_img,calib["fov"],calib["rotation"][2],calib["rotation"][0])
        equ , mask = P.toEquirec(target_h,2*target_h)

            
        

        _fov = 100
        P = Perspective(np.zeros((rgb_img.shape[0],rgb_img.shape[1],rgb_img.shape[2])),_fov,calib["rotation"][2],calib["rotation"][0])
        _ , mask = P.toEquirec(target_h,2*target_h)
        # Smooth the mask with a large Gaussian kernel to create a fading effect
        mask = cv.GaussianBlur(np.float32(mask), (99,99), 0)
        #mask = cv.medianBlur(np.float32(mask),99)
        mask = np.repeat(mask[:, :, np.newaxis], equ.shape[-1], axis=2)



        cam_mat = o3d.camera.PinholeCameraIntrinsic(cols, rows, fx,  fy, cx, cy)
        K = cam_mat.intrinsic_matrix.astype(np.float32)
        

        normalized = depth_to_array(depth)
        in_meters = 1000 * normalized


        cam_points = np.zeros((rows,cols, 3))
        i = 0
        u, v = np.meshgrid(range(cols),range(rows))
        x = (u - cx) * in_meters[v, u] / fx
        y = (v - cy) * in_meters[v, u] / fy
        z = in_meters[v, u]
        cam_points = np.stack([x,y,z], axis=-1)#.reshape(-1,3)
        
        #range_img  = np.linalg.norm(np.stack([x,y,z], axis=-1),axis=-1)

        # cam_points_ = rotate_point_cloud(cam_points ,calib["rotation"][2],calib["rotation"][0]).reshape(-1,3)
        # x = cam_points_[...,0]
        # y = cam_points_[...,1]
        # z = cam_points_[...,2]
        # cam_points = np.stack([x,z,-y], axis=-1)#
        
        
        #xyz = np.concatenate(points_list,axis=0)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cam_points.reshape(-1,3))
        pcd.colors = o3d.utility.Vector3dVector(np.float32(semantic.reshape(-1,3)/255.0))
        
        R = pcd.get_rotation_matrix_from_axis_angle(np.deg2rad((calib["rotation"][0], calib["rotation"][2], 0)))
        pcd.rotate(R, center=(0, 0, 0))
        
        pcd_comb += pcd
        
        P = Perspective(instance,calib["fov"],calib["rotation"][2],calib["rotation"][0], interpolation=cv.INTER_NEAREST)
        equ_instance, mask_instance = P.toEquirec(target_h,2*target_h)
        
        P = Perspective(semantic,calib["fov"],calib["rotation"][2],calib["rotation"][0], interpolation=cv.INTER_NEAREST)
        equ_semantic, _ = P.toEquirec(target_h,2*target_h)
        mask_instance = np.repeat(mask_instance[:, :, np.newaxis], equ.shape[-1], axis=2)
        
        if cam_num == 0:
            equ_stitched = equ
            #equ_range_stitched = equ_range
            equ_instance_stitched = equ_instance
            equ_semantic_stitched = equ_semantic
        else:
            equ_stitched = mask * equ + (1-mask)*equ_stitched
            #equ_range_stitched = mask_instance[...,0] * equ_range + (1-mask_instance[...,0])*equ_range_stitched
            equ_instance_stitched = mask_instance*equ_instance + (1-mask_instance) * equ_instance_stitched
            equ_semantic_stitched = mask_instance*equ_semantic + (1-mask_instance) * equ_semantic_stitched
            
    
    
    pcd_comb_ = copy.deepcopy(pcd_comb)
    #pose_img = project_poses(label_file,equ_stitched, pcd_comb)
    xyz = np.asarray(pcd_comb_.points).reshape(-1,3)
    
    idx = np.where(np.linalg.norm(xyz,axis=-1) <= 100)
    
    pcd_comb_.points = o3d.utility.Vector3dVector(xyz[idx].reshape(-1,3))
    pcd_comb_.colors = o3d.utility.Vector3dVector(xyz[idx].reshape(-1,3))
    
    # rotate to make y facing up
    R = pcd_comb_.get_rotation_matrix_from_axis_angle((0, np.pi/2, 0))
    pcd_comb_.rotate(R, center=(0, 0, 0))

    R = pcd_comb_.get_rotation_matrix_from_axis_angle((-np.pi/2, 0, 0))
    pcd_comb_.rotate(R, center=(0, 0, 0))
    
    R = pcd_comb_.get_rotation_matrix_from_axis_angle((0, 0, 0))
    pcd_comb_.rotate(R, center=(0, 0, 0))
    
    

    xyz = np.asarray(pcd_comb_.points)
    xyz_img, _, _, (phi_min, phi_max) = spherical_projection(xyz,height=target_h,width=2*target_h,theta_range=[-np.pi/2, np.pi/2])
    
    xyz[:,1] = -xyz[:,1]
    pcd_comb_.points = o3d.utility.Vector3dVector(xyz)
    
    bboxes = project_poses(label_file,np.uint8(equ_stitched), pcd_comb_)
    

    range_img  = np.linalg.norm(xyz_img,axis=-1)

    # cv.imshow("equ", np.uint8(equ_stitched)[::2,::2,:])  
    # cv.imshow("range_preview", (np.minimum(range_img,100)/100.0)[::2,::2])  
    
    # cv.imshow("rel_height_change", cv.applyColorMap(np.uint8(255*(5+np.maximum(np.minimum(xyz_img[...,2],5),-5)/10)[::2,::2]), cv.COLORMAP_CIVIDIS))  
    # cv.waitKey(0)
    

    equi_save_path = "/workspace/data/CARLA_HIGH_RES_LIDAR/Ladybug5_0004/equirectangular"
    os.makedirs(equi_save_path, exist_ok=True)

    cv.imwrite(os.path.join(equi_save_path,"rgb_%04d.png" %(frame_id)), equ_stitched)
    cv.imwrite(os.path.join(equi_save_path,"labels_%04d.png" %(frame_id)), equ_instance_stitched)
    cv.imwrite(os.path.join(equi_save_path,"semantic_%04d.png" %(frame_id)), equ_semantic_stitched)
    cv.imwrite(os.path.join(equi_save_path,"xyz_img_%04d.exr" %(frame_id)), xyz_img.astype(np.float32))
    with open(os.path.join(equi_save_path,"annotations_%04d.json" %(frame_id)), 'w') as f:
        json.dump(bboxes, f, cls=NumpyEncoder)
