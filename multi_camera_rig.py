import glob
import os
import sys
import argparse

from datetime import datetime
import random
import numpy as np
from matplotlib import cm

import json
import logging

import queue
import threading
import copy

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def get_lidar_point(loc, w2s):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_sensor = np.dot(w2s, point)

    # New we must change from UE4's coordinate system
    #point_sensor[:, :1] = -point_sensor[:, :1]
    # and we remove the fourth componebonent also
    return point_sensor[0:3]

def get_all_static_actors(world):
    # get the blueprint library
    #blueprint_library = world.get_blueprint_library()
    labels = {}
    vehicle_actors = world.get_environment_objects(carla.CityObjectLabel.Car)
    vehicle_actors += world.get_environment_objects(carla.CityObjectLabel.Bicycle)
    vehicle_actors += world.get_environment_objects(carla.CityObjectLabel.Bus)
    vehicle_actors += world.get_environment_objects(carla.CityObjectLabel.Truck)
    vehicle_actors += world.get_environment_objects(carla.CityObjectLabel.Motorcycle)
    for npc in vehicle_actors:

        npc_id = npc.id
        npc_type = npc.type
        npc_name = npc.name
        bb = npc.bounding_box
        ex = bb.extent.x
        ey = bb.extent.y
        ez = bb.extent.z
        x = bb.location.x
        y = bb.location.y
        z = bb.location.z
        pitch = bb.rotation.pitch
        yaw = bb.rotation.yaw
        roll = bb.rotation.roll

        npc_dict = {}
        if "Harley" in npc_name:
            npc_dict["number_of_wheels"] = 2
        elif "Kawasaki" in npc_name:
            npc_dict["number_of_wheels"] = 2
        elif "Bike" in npc_name:
            npc_dict["number_of_wheels"] = 2
        elif "Vespa" in npc_name:
            npc_dict["number_of_wheels"] = 2
        elif "Yamaha" in npc_name:
            npc_dict["number_of_wheels"] = 2
        else:
            npc_dict["number_of_wheels"] = 4
        npc_dict["motion_state"] = "static"
        npc_dict["velocity"] = [0.0,0.0,0.0]
        npc_dict["acceleration"] = [0.0,0.0,0.0]
        npc_dict["semantic_tag"] = npc_type
        npc_dict["type_id"] = npc_name
        npc_dict["extent"] = [ex,ey,ez]
        npc_dict["location"] = [x,y,z]
        npc_dict["rotation"] = [pitch,yaw,roll]


        #npc_dict["verts"] = [[v.x, v.y, v.z] for v in bb.get_world_vertices(npc.get_transform())]
        labels[npc_id] = npc_dict
    return labels

def get_lidar_and_labels(world, static_labels):


    #json.dump(meta, open(os.path.join(calib_path,'%.6d.json' % frame_id), 'w' ) )
    # first get all actors and assume them as static
    labels = static_labels #get_all_static_actors(world)
    #labels = {}
    # get all dynamic walkeras and vehicles as actors
    vehicle_actors = world.get_actors().filter('*vehicle.*')
    walker_actors = world.get_actors().filter('*walker.*')
    actors = list(vehicle_actors) + list(walker_actors)
    
    # add dynamic actors, overwrite existing actors with static
    for npc in actors:
        transform = npc.get_transform()
        rotation = transform.rotation
        location = transform.location
        npc_id = npc.id
        
        npc_semantic_tag = npc.semantic_tags
        
        npc_type_id = npc.type_id
        npc_velocity = npc.get_velocity()
        npc_acceleration = npc.get_acceleration()
        bb = npc.bounding_box
        
        ex = bb.extent.x
        ey = bb.extent.y
        ez = bb.extent.z
        x = location.x + bb.location.x
        y = location.y + bb.location.y
        z = location.z + bb.location.z
        pitch = rotation.pitch + bb.rotation.pitch
        yaw = rotation.yaw + bb.rotation.yaw
        roll = rotation.roll + bb.rotation.roll

        vx = npc_velocity.x
        vy = npc_velocity.y
        vz = npc_velocity.z
        ax = npc_acceleration.x
        ay = npc_acceleration.y
        az = npc_acceleration.z
        
        npc_dict = {}
        if isinstance(npc, carla.Walker):
            bones = npc.get_bones()
            # prepare the bones (get name and world position)
            boneIndex = {}  
            for i, bone in enumerate(bones.bone_transforms):
                boneIndex[bone.name] = {"world": [bone.world.location.x, bone.world.location.y, bone.world.location.z]}
            npc_dict["bones"] = boneIndex
        # if isinstance(npc, carla.Vehicle):
        #     bp = blueprint_library.find(npc_type_id)
        #     npc_dict["number_of_wheels"] = int(bp.get_attribute('number_of_wheels'))
        npc_dict["motion_state"] = "dynamic"
        npc_dict["velocity"] = [vx,vy,vz]
        npc_dict["acceleration"] = [ax,ay,az]
        npc_dict["extent"] = [ex,ey,ez]
        npc_dict["location"] = [x,y,z]
        npc_dict["rotation"] = [pitch,yaw,roll]
        npc_dict["semantic_tag"] = npc_semantic_tag
        npc_dict["type_id"] = npc_type_id
        
        npc_dict["verts"] = [[v.x, v.y, v.z] for v in bb.get_world_vertices(npc.get_transform())]
        labels[npc_id] = npc_dict
    return labels

def save_lidar_and_labels(labels, frame_id, save_path):

    labels_path = os.path.join(save_path, "labels")

    os.makedirs(labels_path, exist_ok=True)


    json.dump(labels, open(os.path.join(labels_path,'%.4d.json' % frame_id), 'w' ) )


# -------------
# Build Sensor Blueprints
# -------------
def sensor_callback(sensor_data, sensor_queue):
    sensor_queue.put(sensor_data)
    
class Sensor:
    initial_ts = 0.0
    initial_loc = carla.Location()
    initial_rot = carla.Rotation()

    def __init__(self, vehicle, world, folder_output, transform):
        self.queue = queue.Queue()
        self.world = world
        self.bp = self.set_attributes(world.get_blueprint_library())
        self.sensor = world.spawn_actor(self.bp, transform, attach_to=vehicle)
        self.sensor.listen(lambda data: sensor_callback(data, self.queue))
        self.sensor_id = self.__class__.sensor_id_glob
        self.__class__.sensor_id_glob += 1
        self.folder_output = folder_output
        self.ts_tmp = 0

def push_snapshot(snapshot):
    snapshots.put(snapshot)

class Labels:
    initial_ts = 0.0

    def __init__(self, world, folder_output):
        self.sensor_frame_id = 0
        self.queue = queue.Queue()
        #self.bp = self.set_attributes(world.get_blueprint_library())
        self.static_labels = get_all_static_actors(world)
        self.world = world
        self.folder_output = folder_output
        self.world.on_tick(lambda ws: self.queue.put(world))
    
    def save(self):
        while not self.queue.empty():
            world = self.queue.get()
            labels = get_lidar_and_labels(world, self.static_labels)
            save_lidar_and_labels(labels, self.sensor_frame_id, self.folder_output)
            self.sensor_frame_id += 1
            
    def dequeue(self):
        while not self.queue.empty():
            data = self.queue.get()


class Camera(Sensor):
    def __init__(self, vehicle, world, folder_output, transform, meta):
        Sensor.__init__(self, vehicle, world, folder_output, transform)
        self.sensor_frame_id = 0
        self.meta = meta
        self.frame_output = self.folder_output+"/images_%s" %str.lower(self.__class__.__name__)
        self.calib_output = self.folder_output+"/calib_%s" %str.lower(self.__class__.__name__)
        
        os.makedirs(self.frame_output) if not os.path.exists(self.frame_output) else [os.remove(f) for f in glob.glob(self.frame_output+"/*") if os.path.isfile(f)]
        os.makedirs(self.calib_output) if not os.path.exists(self.calib_output) else [os.remove(f) for f in glob.glob(self.calib_output+"/*") if os.path.isfile(f)]

        with open(self.folder_output+"/full_ts_camera.txt", 'w') as file:
            file.write("# frame_id timestamp\n")

        print('created %s' % self.sensor)

    def save(self, color_converter=carla.ColorConverter.Raw):
        first_call = True
        while not self.queue.empty():
            meta = self.meta
            data = self.queue.get()
            # Todo save world to sensor for every sensor
            if self.sensor_id == 0:
                #save_lidar_and_labels(self.world, self.sensor_frame_id, self.folder_output)
                with open(self.folder_output+"/full_ts_camera.txt", 'a') as file:
                    file.write(str(self.sensor_frame_id)+" "+str(data.timestamp - Sensor.initial_ts)+"\n")
            if first_call:
                first_call = False
                
                weather =  self.world.get_weather()
                print("weather", weather)
                w2s = np.array(data.transform.get_matrix())
                meta["world2sensor"] = w2s.tolist()
                meta["fov"] = data.fov
                meta["height"] = data.height
                meta["width"] = data.width
                


                ts = data.timestamp-Sensor.initial_ts
                # if ts - self.ts_tmp > 0.6 or (ts - self.ts_tmp) < 0: #check for 2Hz camera acquisition
                #     print("[Error in timestamp] Camera: previous_ts %f -> ts %f" %(self.ts_tmp, ts))
                #     sys.exit()
                self.ts_tmp = ts

                file_path = self.frame_output+"/%04d_%d.png" %(self.sensor_frame_id, self.sensor_id)
                file_path_calib = self.calib_output+"/%04d_%d.json" %(self.sensor_frame_id, self.sensor_id)
                json.dump(meta, open(file_path_calib, 'w' ) )
                x = threading.Thread(target=data.save_to_disk, args=(file_path, color_converter))
                x.start()
                print("Export : "+file_path)
                
                
                self.sensor_frame_id += 1

    def dequeue(self):
        while not self.queue.empty():
            data = self.queue.get()
        

class RGB(Camera):
    sensor_id_glob = 0
    def __init__(self, vehicle, world, folder_output, transform, meta):
        Camera.__init__(self, vehicle, world, folder_output, transform, meta)

    def set_attributes(self, blueprint_library):
        camera_bp = blueprint_library.find('sensor.camera.rgb')

        # Ladybug5
        camera_bp.set_attribute('image_size_x', '3072')
        camera_bp.set_attribute('image_size_y', '3072')
        camera_bp.set_attribute('fov', '120') #120 degrees # Always fov on width even if width is smaller than height
        camera_bp.set_attribute('enable_postprocess_effects', 'True')
        camera_bp.set_attribute('sensor_tick', '0.0') # 2Hz camera
        camera_bp.set_attribute('gamma', '2.2')
        camera_bp.set_attribute('motion_blur_intensity', '0')
        camera_bp.set_attribute('motion_blur_max_distortion', '0')
        camera_bp.set_attribute('motion_blur_min_object_screen_size', '0')
        camera_bp.set_attribute('shutter_speed', '1000') #1 ms shutter_speed
        camera_bp.set_attribute('lens_k', '0')
        camera_bp.set_attribute('lens_kcube', '0')
        camera_bp.set_attribute('lens_x_size', '0')
        camera_bp.set_attribute('lens_y_size', '0')
        return camera_bp
    
    def save(self):
        Camera.save(self)
        
    def dequeue(self):
        Camera.dequeue(self)

class SS(Camera):
    sensor_id_glob = 10
    def __init__(self, vehicle, world, folder_output, transform, meta):
        Camera.__init__(self, vehicle, world, folder_output, transform, meta)

    def set_attributes(self, blueprint_library):
        camera_ss_bp = blueprint_library.find('sensor.camera.semantic_segmentation')

        # Ladybug5
        camera_ss_bp.set_attribute('image_size_x', '3072')
        camera_ss_bp.set_attribute('image_size_y', '3072')
        camera_ss_bp.set_attribute('fov', '120') #120 degrees # Always fov on width even if width is smaller than height
        camera_ss_bp.set_attribute('sensor_tick', '0.0')  # 2Hz camera
        return camera_ss_bp

    def save(self):
        Camera.save(self, color_converter=carla.ColorConverter.CityScapesPalette)
        
    def dequeue(self):
        Camera.dequeue(self)
        
class IS(Camera):
    sensor_id_glob = 20
    def __init__(self, vehicle, world, folder_output, transform, meta):
        Camera.__init__(self, vehicle, world, folder_output, transform, meta)

    def set_attributes(self, blueprint_library):
        camera_ss_bp = blueprint_library.find('sensor.camera.instance_segmentation')

        # Ladybug5
        camera_ss_bp.set_attribute('image_size_x', '3072')
        camera_ss_bp.set_attribute('image_size_y', '3072')
        camera_ss_bp.set_attribute('fov', '120') #120 degrees # Always fov on width even if width is smaller than height
        camera_ss_bp.set_attribute('sensor_tick', '0.0')  # 2Hz camera
        return camera_ss_bp

    def save(self):
        Camera.save(self)
    
    def dequeue(self):
        Camera.dequeue(self)
        
class Depth(Camera):
    sensor_id_glob = 30
    #sensor_id_glob = 0
    def __init__(self, vehicle, world, folder_output, transform, meta):
        Camera.__init__(self, vehicle, world, folder_output, transform, meta)

    def set_attributes(self, blueprint_library):
        camera_ss_bp = blueprint_library.find('sensor.camera.depth')

        # Ladybug5
        camera_ss_bp.set_attribute('image_size_x', '3072')
        camera_ss_bp.set_attribute('image_size_y', '3072')
        camera_ss_bp.set_attribute('fov', '120') #120 degrees # Always fov on width even if width is smaller than height
        camera_ss_bp.set_attribute('sensor_tick', '0.0')  # 2Hz camera
        return camera_ss_bp

    def save(self):
        Camera.save(self, color_converter=carla.ColorConverter.Raw)

    def dequeue(self):
        Camera.dequeue(self)
        

class Ladybug5:
    def __init__(self, vehicle, world, folder_output, transform, meta):
        # build rgb camera
        self.list_rgb_cam = []
        for i_cam in range(5):
            meta_ = copy.deepcopy(meta)
            rotation = carla.Rotation(pitch=transform.rotation.pitch, yaw=transform.rotation.yaw+72*i_cam, roll=transform.rotation.roll)
            meta_["rotation"] = np.array([rotation.pitch, rotation.roll, rotation.yaw]).tolist()
            cam_transform = carla.Transform(transform.location, carla.Rotation(pitch=transform.rotation.pitch, yaw=transform.rotation.yaw+72*i_cam, roll=transform.rotation.roll))
            self.list_rgb_cam.append(RGB(vehicle, world, folder_output, cam_transform, meta_))
        # build up facing camera
        meta_ = copy.deepcopy(meta)
        rotation = carla.Rotation(pitch=transform.rotation.pitch+90, yaw=transform.rotation.yaw, roll=transform.rotation.roll)
        meta_["rotation"] = np.array([rotation.pitch, rotation.roll, rotation.yaw]).tolist()
        cam_transform = carla.Transform(transform.location, carla.Rotation(pitch=transform.rotation.pitch+90, yaw=transform.rotation.yaw, roll=transform.rotation.roll))
        self.list_rgb_cam.append(RGB(vehicle, world, folder_output, cam_transform, meta_))
        
        # build down facing camera
        meta_ = copy.deepcopy(meta)
        rotation = carla.Rotation(pitch=transform.rotation.pitch-90, yaw=transform.rotation.yaw, roll=transform.rotation.roll)
        meta_["rotation"] = np.array([rotation.pitch, rotation.roll, rotation.yaw]).tolist()
        cam_transform = carla.Transform(transform.location, carla.Rotation(pitch=transform.rotation.pitch-90, yaw=transform.rotation.yaw, roll=transform.rotation.roll))
        self.list_rgb_cam.append(RGB(vehicle, world, folder_output, cam_transform, meta_))
        RGB.sensor_id_glob = 0
        
        # # build instance camera
        self.list_is_cam = []
        for i_cam in range(5):
            meta_ = copy.deepcopy(meta)
            rotation = carla.Rotation(pitch=transform.rotation.pitch, yaw=transform.rotation.yaw+72*i_cam, roll=transform.rotation.roll)
            meta_["rotation"] = np.array([rotation.pitch, rotation.roll, rotation.yaw]).tolist()
            cam_transform = carla.Transform(transform.location, carla.Rotation(pitch=transform.rotation.pitch, yaw=transform.rotation.yaw+72*i_cam, roll=transform.rotation.roll))
            self.list_is_cam.append(IS(vehicle, world, folder_output, cam_transform, meta_))

        meta_ = copy.deepcopy(meta)
        rotation = carla.Rotation(pitch=transform.rotation.pitch+90, yaw=transform.rotation.yaw, roll=transform.rotation.roll)
        meta_["rotation"] = np.array([rotation.pitch, rotation.roll, rotation.yaw]).tolist()
        cam_transform = carla.Transform(transform.location, carla.Rotation(pitch=transform.rotation.pitch+90, yaw=transform.rotation.yaw, roll=transform.rotation.roll))
        self.list_is_cam.append(IS(vehicle, world, folder_output, cam_transform, meta_))
        
        meta_ = copy.deepcopy(meta)
        rotation = carla.Rotation(pitch=transform.rotation.pitch-90, yaw=transform.rotation.yaw, roll=transform.rotation.roll)
        meta_["rotation"] = np.array([rotation.pitch, rotation.roll, rotation.yaw]).tolist()
        cam_transform = carla.Transform(transform.location, carla.Rotation(pitch=transform.rotation.pitch-90, yaw=transform.rotation.yaw, roll=transform.rotation.roll))
        self.list_is_cam.append(IS(vehicle, world, folder_output, cam_transform, meta_))
        IS.sensor_id_glob = 0
        
        # build semantic camera
        self.list_ss_cam = []
        for i_cam in range(5):
            meta_ = copy.deepcopy(meta)
            rotation = carla.Rotation(pitch=transform.rotation.pitch, yaw=transform.rotation.yaw+72*i_cam, roll=transform.rotation.roll)
            meta_["rotation"] = np.array([rotation.pitch, rotation.roll, rotation.yaw]).tolist()
            cam_transform = carla.Transform(transform.location, carla.Rotation(pitch=transform.rotation.pitch, yaw=transform.rotation.yaw+72*i_cam, roll=transform.rotation.roll))
            self.list_ss_cam.append(SS(vehicle, world, folder_output, cam_transform, meta_))
        
        meta_ = copy.deepcopy(meta)
        rotation = carla.Rotation(pitch=transform.rotation.pitch+90, yaw=transform.rotation.yaw, roll=transform.rotation.roll)
        meta_["rotation"] = np.array([rotation.pitch, rotation.roll, rotation.yaw]).tolist()
        cam_transform = carla.Transform(transform.location, carla.Rotation(pitch=transform.rotation.pitch+90, yaw=transform.rotation.yaw, roll=transform.rotation.roll))
        self.list_ss_cam.append(SS(vehicle, world, folder_output, cam_transform, meta_))
        
        meta_ = copy.deepcopy(meta)
        rotation = carla.Rotation(pitch=transform.rotation.pitch-90, yaw=transform.rotation.yaw, roll=transform.rotation.roll)
        meta_["rotation"] = np.array([rotation.pitch, rotation.roll, rotation.yaw]).tolist()
        cam_transform = carla.Transform(transform.location, carla.Rotation(pitch=transform.rotation.pitch-90, yaw=transform.rotation.yaw, roll=transform.rotation.roll))
        self.list_ss_cam.append(SS(vehicle, world, folder_output, cam_transform, meta_))
        SS.sensor_id_glob = 0
        
        
        
        # build depth camera
        self.list_depth_cam = []
        for i_cam in range(5):
            meta_ = copy.deepcopy(meta)
            rotation = carla.Rotation(pitch=transform.rotation.pitch, yaw=transform.rotation.yaw+72*i_cam, roll=transform.rotation.roll)
            meta_["rotation"] = np.array([rotation.pitch, rotation.roll, rotation.yaw]).tolist()
            cam_transform = carla.Transform(transform.location, carla.Rotation(pitch=transform.rotation.pitch, yaw=transform.rotation.yaw+72*i_cam, roll=transform.rotation.roll))
            self.list_depth_cam.append(Depth(vehicle, world, folder_output, cam_transform, meta_))

        meta_ = copy.deepcopy(meta)
        rotation = carla.Rotation(pitch=transform.rotation.pitch+90, yaw=transform.rotation.yaw, roll=transform.rotation.roll)
        meta_["rotation"] = np.array([rotation.pitch, rotation.roll, rotation.yaw]).tolist()
        cam_transform = carla.Transform(transform.location, carla.Rotation(pitch=transform.rotation.pitch+90, yaw=transform.rotation.yaw, roll=transform.rotation.roll))
        self.list_depth_cam.append(Depth(vehicle, world, folder_output, cam_transform, meta_))
        
        meta_ = copy.deepcopy(meta)
        rotation = carla.Rotation(pitch=transform.rotation.pitch-90, yaw=transform.rotation.yaw, roll=transform.rotation.roll)
        meta_["rotation"] = np.array([rotation.pitch, rotation.roll, rotation.yaw]).tolist()
        cam_transform = carla.Transform(transform.location, carla.Rotation(pitch=transform.rotation.pitch-90, yaw=transform.rotation.yaw, roll=transform.rotation.roll))
        self.list_depth_cam.append(Depth(vehicle, world, folder_output, cam_transform, meta_))
        Depth.sensor_id_glob = 0

    def save(self):
        for cam in self.list_rgb_cam:
            cam.save()

        for cam in self.list_is_cam:
            cam.save()

        for cam in self.list_depth_cam:
            cam.save()
            
        for cam in self.list_ss_cam:
            cam.save()
    
    def dequeue(self):
        for cam in self.list_rgb_cam:
            cam.dequeue()
        for cam in self.list_is_cam:
            cam.dequeue()
        
        for cam in self.list_depth_cam:
            cam.dequeue()
        
        for cam in self.list_ss_cam:
            cam.dequeue()




## Build World Queue for Metadata and 3D annotations
snapshots = queue.Queue()

def push_snapshot(snapshot):
    snapshots.put(snapshot)

def main(arg):
    args = arg
    """Main function of the script"""
    client = carla.Client(arg.host, arg.port)
    client.set_timeout(50.0)
    synchronous_master = False
    #world = client.get_world()
    
    try:
        world = client.load_world(args.town)
        print(client.get_available_maps())
        vehicles_list = []
        walkers_list = []
        all_id = []
        original_settings = world.get_settings()
        settings = world.get_settings()
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        delta = 1.0/arg.frame_rate

        settings.fixed_delta_seconds = delta
        settings.synchronous_mode = True
        settings.no_rendering_mode = arg.no_rendering
        world.apply_settings(settings)

        blueprints = get_actor_blueprints(world, args.filterv, args.generationv)
        blueprintsWalkers = get_actor_blueprints(world, args.filterw, args.generationw)

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
            blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        hero = args.hero
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # Set automatic vehicle lights update if specified
        if args.car_lights_on:
            all_vehicle_actors = world.get_actors(vehicles_list)
            for actor in all_vehicle_actors:
                traffic_manager.update_vehicle_lights(actor, True)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.05      # how many pedestrians will run
        percentagePedestriansCrossing = 0.3     # how many pedestrians will walk through the road
        if args.seedw:
            world.set_pedestrians_seed(args.seedw)
            random.seed(args.seedw)
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        # if args.asynch or not synchronous_master:
        #     world.wait_for_tick()
        # else:
        #world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # Example of how to use Traffic Manager parameters
        traffic_manager.global_percentage_speed_difference(30.0)

        # -------------
        # Set Weather
        # -------------
        
        #world.set_weather(carla.WeatherParameters.SoftRainSunset)
        
        
        
        # -------------
        # Spawn Recording Vehicle
        # -------------
        
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(arg.filter)[0]
        vehicle_transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
        vehicle.set_autopilot(True)

        # -------------
        # Build Sensor Blueprints
        # -------------
        
        user_offset = carla.Location(arg.x, arg.y, arg.z)
        vehicle_offset = carla.Location(x=0.25, z=2.0)
        mounting_offset = vehicle_offset + user_offset
        rotation = carla.Rotation(pitch=arg.pitch, roll=args.roll, yaw=args.yaw)
        transform = carla.Transform(vehicle_offset + user_offset, rotation)
        
        weather = carla.WeatherParameters(
        cloudiness=0.0,
        precipitation=0.0,
        sun_altitude_angle=45.0)

        world.set_weather(weather)

        

        meta = {}
        meta["vehicle"] = str(arg.filter)
        meta["frame_rate"] = arg.frame_rate
        meta["mounting_offset"] = np.array([mounting_offset.x, mounting_offset.y, mounting_offset.z]).tolist()
        meta["mounting_angle"] = np.array([rotation.pitch, rotation.roll, rotation.yaw]).tolist()

        camera_rig = Ladybug5(vehicle, world, arg.save_path, transform, meta)
        frame = 0
        dt0 = datetime.now()
        n_ticks = 0
        ramp_up_frames = 10
        while frame < arg.n_frames:
            frame_id = world.tick()
            n_ticks += 1
            if (frame_id % args.save_nth_frame == 0) and (n_ticks >= ramp_up_frames):
                # TODO. somhow there is a shift between 
                static_labels = get_all_static_actors(world)
                labels = get_lidar_and_labels(world, static_labels)
                save_lidar_and_labels(labels, frame, arg.save_path)
                camera_rig.save()
                frame += 1
                
            else:
                camera_rig.dequeue()        
            if (frame_id % 10 == 0) and (frame_id > 0):
                tmp_var_ = 1
                # enable fog and rain in 50% of the cases. Sweep from 10% to 90% if enabled
                #weather = carla.WeatherParameters(
                #cloudiness=np.random.choice([10.0* i for i in range(1,6)]+[0.0* i for i in range(0,9)]),
                #precipitation=np.random.choice([10.0* i for i in range(1,9)]+[0.0* i for i in range(0,8)]),
                #precipitation_deposits=np.random.choice([10.0* i for i in range(1,9)]+[0.0* i for i in range(0,8)]),
                #sun_altitude_angle=np.random.choice([10.0* i for i in range(3,6)]),
                #sun_azimuth_angle=np.random.choice([10.0* i for i in range(0,36)]))
                #world.set_weather(weather)
                

    finally:
        print("destroyer")
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)

        vehicle.destroy()



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        help='use the no-rendering mode which will provide some extra'
        ' performance but you will lose the articulated objects in the'
        ' lidar, such as pedestrians')
    argparser.add_argument(
        '--semantic',
        default=True,
        action='store_true',
        help='use the semantic lidar instead, which provides ground truth'
        ' information')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--no-autopilot',
        action='store_false',
        help='disables the autopilot so the vehicle will remain stopped')
    argparser.add_argument(
        '--show-axis',
        action='store_true',
        help='show the cartesian coordinates axis')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='model3',
        help='actor filter (default: "vehicle.*")')

    argparser.add_argument(
        '--n_frames',
        default=10,
        type=int,
        help='number of frames to save (default: 100)')
    argparser.add_argument(
        '--save_nth_frame',
        default=10,
        type=int,
        help='frames to be saved (default: 10')
    argparser.add_argument(
        '-x',
        default=0.0,
        type=float,
        help='offset in the sensor position in the X-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-y',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Y-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-z',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Z-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-pitch',
        default=0.0,
        type=float,
        help='pitch of the sensor in degrees (default: 0.0)')
    argparser.add_argument(
        '-roll',
        default=0.0,
        type=float,
        help='roll of the sensor in degrees (default: 0.0)')
    argparser.add_argument(
        '-yaw',
        default=0.0,
        type=float,
        help='yaw of the sensor in degrees (default: 0.0)')
    argparser.add_argument(
        '-frame_rate',
        default=10.0,
        type=float,
        help='frame rate')
    argparser.add_argument(
        '-save_path',
        default="/tmp/Ladybug5",
        type=str,
        help='save path')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=20,
        type=int,
        help='Number of vehicles (default: 30)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=80,
        type=int,
        help='Number of walkers (default: 10)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument(
        '--generationv',
        metavar='G',
        default='All',
        help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--generationw',
        metavar='G',
        default='2',
        help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--asynch',
        action='store_true',
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument(
        '--seedw',
        metavar='S',
        default=0,
        type=int,
        help='Set the seed for pedestrians module')
    argparser.add_argument(
        '--pedestrians_cross_factor',
        metavar='S',
        default=0.05,
        type=float,
        help='Sets the percentage of pedestrians that can walk on the road or cross at any point on the road. Value should be between 0.0 and 1.0. For example, a value of 0.1 would allow 10% of pedestrians to walk on the road. Default is 0.0')
    argparser.add_argument(
        '--town',
        metavar='S',
        default="Town10HD",
        type=str,
        help='Set the town')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=True,
        help='Enable automatic car light management')
    argparser.add_argument(
        '--hero',
        action='store_true',
        default=False,
        help='Set one of the vehicles as hero')
    argparser.add_argument(
        '--respawn',
        action='store_true',
        default=False,
        help='Automatically respawn dormant vehicles (only in large maps)')

    args = argparser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')
