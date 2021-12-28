# Speed Violation Detection
Real-time Speed violation detection using Yolo and Deep-Sort
## Instructions
1. First clone the repository
2. Run the conda-env.yml to install a dedicated conda environment and install all the necessary repositories.
```
conda env create -f conda-env.yml
conda activate yolov4-gpu
```
3. If you don't have a gpu, comment out 8 and 9 lines and remove '-gpu' segment in 11 th line. (tensorflow==2.3.0 instead of tensorflow-gpu==2.3.0)
4. Please note that you need a comparatively powerful gpu in order to get a descent result.
3. Download yolov4 pre-trained weights --> https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights 
4. Move the yolov4.weights file to /data folder
5. Excute following command to convert darknet weights to tensorflow model.
```
python save_model.py --model yolov4 
```
5. Run this commad to test the program
```
python object_tracker_speed_violation.py --output ./outputs/processed_vids/speed.avi --model yolov4
```


This is the output you will get if you successfuly ran the program;

![Alt text](output.gif) 
