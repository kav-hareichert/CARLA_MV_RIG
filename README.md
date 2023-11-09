# CARLA_MV_RIG
A multi view rig in CARLA to record 360° sequences for color cameras, instance segmentation and depth estimation.

## Setup
1. Install CARLA 9.14 via Docker
```bash
docker pull carlasim/carla:0.9.14
```
See https://carla.readthedocs.io/en/latest/build_docker/ for more details

2. Install the CARLA Python API
See https://carla.readthedocs.io/en/0.9.14/start_quickstart/ for more details

## Recording
For a simple recording run:
```bash
python multi_camera_rig.py
```
We enable various options for simulation as args. This includes the map used, the weather, the traffic composition, the sensor placement, and many more options.
Make sure to set a valid save_path!

## Post Processing
We stitch the panoramas in a postprocessing step.
For the stitching run:
```bash
python Stitch_CARLA_Panorama.py -data_path "path/to/your/recording
```
The stitching will create an additional folder "equirectangular" which contains the stitched panoramas for RGB, range, semantic segmentation, and instance segmentation.

## Data
The final data is stored in the following folder structure
```bash
├── calib_depth # calibration intrinsics and world2sensor
├── calib_is # calibration intrinsics and world2sensor
├── calib_rgb # calibration intrinsics and world2sensor
├── calib_ss # calibration intrinsics and world2sensor
├── equirectangular # panoramas
│   ├── labels_0000.png # instance
│   ├── labels_*.png
│   ├── prange_0000.jpg # preview range
│   ├── prange_*.jpg
│   ├── range_0000.png # raw range
│   ├── range_*.png
│   ├── rgb_0000.png # rgb
│   ├── rgb_*.png
│   ├── semantic_0000.png # semantic
│   └── semantic_*.png
├── full_ts_camera.txt
├── images_depth # six depth images per time step
├── images_is # six instance images per time step
├── images_rgb # six rgb images per time step
│   ├── 0000_0.png
│   ├── 0000_1.png
│   ├── 0000_2.png
│   ├── 0000_3.png
│   ├── 0000_4.png
│   ├── 0000_5.png
│   ├── 0000_6.png
│   ├── 0009_0.png
│   ├── *_1.png
│   ├── *_2.png
│   ├── *_3.png
│   ├── *_4.png
│   ├── *_5.png
│   └── *_6.png
├── images_ss # six semantic images per time step
│   ├── 0000_10.png
│   ├── 0000_11.png
│   ├── 0000_12.png
│   ├── 0000_13.png
│   ├── 0000_14.png
│   ├── 0000_15.png
│   ├── 0000_16.png
│   ├── *_10.png
│   ├── *_11.png
│   ├── *_12.png
│   ├── *_13.png
│   ├── *_14.png
│   ├── *_15.png
│   └── *_16.png
└── labels # Labels are in world coordinates
    ├── 0000.json
    └── *.json
```

![rgbImage](images/rgb_0000.png)

