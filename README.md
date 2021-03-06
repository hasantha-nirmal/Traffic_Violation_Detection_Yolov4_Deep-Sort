# Traffic Violation Detection
Real-time Speed violation detection and Lane-Line violation detection using Yolov4 and Deep-Sort
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
5. Run this commad to test the speed violation detection program
```
python object_tracker_speed_violation.py --output ./outputs/processed_vids/speed.avi --model yolov4
```
6. Run this commad to test the Lane-Line violation detection program after changing the paths in lane_line_extract3.py (5th line, change according to your PC path)
```
python lane_line_viol_detection.py --output ./outputs/processed_vids/lane.avi --model yolov4
```
7. Run this commad to test the Parking violation detection program
```
python parking_violation_detection.py  --output ./outputs/processed_vids/parking.avi --model yolov4
```


<!-- This is the output you will get if you successfuly ran the speed violation detection program; -->

<!-- <img src="output.gif" width="400" height="200"/> -->

## If you looking for a better explaination of the main python script, check out my medium article;
## Speed Violation Detection
https://medium.com/@hasanthakdu/speed-violation-detection-of-real-time-video-footage-using-computer-vision-8bab662cf8b4

## Lane Line Violation Detection
https://medium.com/@hasanthakdu/lane-line-violation-detection-using-yolo-and-deep-sort-f19774b3c739

## Parking Violation Detetcion
https://medium.com/@hasanthakdu/parking-violation-detection-system-using-computer-vision-3ee784cb5d43


## Big Shout out to these developers/heroes for buiding the backbone of this whole project
https://github.com/theAIGuysCode/yolov4-deepsort

https://github.com/hunglc007/tensorflow-yolov4-tflite

https://github.com/nwojke/deep_sort

