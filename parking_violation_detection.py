import os

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from datetime import datetime
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
from skimage.draw import line
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import xlwt

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # open existing workbook
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("Sheet1")

    # Specifying style
    style = xlwt.easyxf('font: bold 1')
    row_num = 0

    cache_matrix = []
    viol_matrix = []
    frame_matrix = []

    # Comapare 2 regions:
    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))

    # Check the immobility of vehicle
    def immobile(bbox, previous_bbox_str):
        previous_bbox0 = int(previous_bbox_str[1:5])
        previous_bbox1 = int(previous_bbox_str[9:13])
        previous_bbox2 = int(previous_bbox_str[17:21])
        previous_bbox3 = int(previous_bbox_str[25:29])

        total = abs(bbox[0] - previous_bbox0) + abs(bbox[1] - previous_bbox1) + abs(bbox[2] - previous_bbox2) + abs(
            bbox[3] - previous_bbox3)
        if total <= 4:
            return True
        else:
            return False

    # Parking space coordinates;
    parking_co = []

    #blanked = np.zeros((658,1024), dtype=np.uint8)
    blanked = np.zeros((2048, 1024), dtype=np.uint8)
    #pts = np.array(([156, 704], [2, 893], [476, 932], [270, 708]))
    pts = np.array(([513, 716], [321, 943], [884, 979], [630, 701]))
    #blanked = np.zeros((720,1280), dtype=np.uint8)
    #pts = np.array(([38, 433], [95, 322], [1246, 570], [1065, 709]))
    cv2.fillPoly(blanked, np.int32([pts]), 255)

    x_cord = np.where(blanked == 255)[1]
    y_cord = np.where(blanked == 255)[0]

    for q in range(0, len(x_cord)):
        parking_co.append((x_cord[q], y_cord[q]))

    # begin video capture
    # try:
    vid = cv2.VideoCapture('data/video/kalutara_park.mp4')
    #vid = cv2.VideoCapture('/home/hasantha/Documents/yolov4-deepsort-master/parking_violation/parking_viol3.mp4' )
    # except:
    # vid = cv2.VideoCapture(video_path)

    #out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #frame = frame[0:1151, 365:1023]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

            # equalize the histogram of the Y channel
            frame[:, :, 0] = cv2.equalizeHist(frame[:, :, 0])

            # convert the YUV image back to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # frame = cv2.resize(frame, (720, 360), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num += 1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['car', 'truck', 'motorbike', 'bicycle']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                        (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            t = int(str(track.track_id))

            # bottom_start_co = (int(bbox[0]),int(bbox[3]))
            # bottom_stop_co = (int(bbox[2]),int(bbox[3]))
            # bbox_bottom_line_co = list(zip(*line(*bottom_start_co, *bottom_stop_co)))

            bbox_bottom_line_co = list(zip(*line(*(int(bbox[0])+50,int(bbox[3])), *(int(bbox[2])-50,int(bbox[3])))))

            # b_box_co = []
            #
            # # Change accordingly
            # X, Y = np.mgrid[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])]
            # for m in range(0, len(X)):
            #     for n in range(0, len(X[0])):
            #         b_box_co.append((X[m][n], Y[m][n]))

            if len(intersection(parking_co, bbox_bottom_line_co)) > 0:
                frame_matrix.append((str(frame_num) + class_name + str(t),
                                     (
                                     str(int(bbox[0])).zfill(4), str(int(bbox[1])).zfill(4), str(int(bbox[2])).zfill(4),
                                     str(int(bbox[3])).zfill(4))))
                chk_index = str(frame_matrix).find(str(frame_num - 1) + class_name + str(t))
                # print(frame_matrix)
                if bool(chk_index + 1) == True:
                    previous_bbox_co_str = str(frame_matrix)[
                                           (chk_index - 1) + len(str(frame_num - 1)) + len(class_name + str(t)) + 5:(chk_index - 1) + len(str(frame_num - 1)) + len(class_name + str(t)) + 5 + 30]


                    # print(previous_bbox_co_str)

                    # To check the mobility
                    if immobile(bbox, previous_bbox_co_str) == True:
                        if str((class_name + str(t))) not in str(cache_matrix):
                            t_start = datetime.now()
                            cache_matrix.append((str(t_start), class_name + str(t)))
                            print(cache_matrix)

                        if (str((class_name + str(t))) in str(cache_matrix)) and (
                                str((class_name + str(t))) not in str(viol_matrix)):

                            index = (str(cache_matrix).find(str((class_name + str(t)))))
                            t_start_cm = str(cache_matrix)[index - 28:index - 11]
                            t_spending = (datetime.now() - datetime.strptime(t_start_cm,
                                                                             '%y-%m-%d %H:%M:%S')).total_seconds()
                            cv2.putText(frame,
                                        class_name + "-" + str(track.track_id) + "waited_time : " + str(t_spending),
                                        (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75, (255, 0, 0), 2)

                            print(class_name + str(t), t_start_cm, t_spending)
                            if t_spending > 10:
                                sheet.write(row_num, 0, str(t_start_cm), style)
                                # sheet.write(row_num, 1, str(round(t_spending, 2)), style)
                                sheet.write(row_num, 1, str(class_name) + str(t), style)
                                row_num += 1
                                workbook.save('outputs/xlsx/parking/details.xls')

                                viol_matrix.append((str((class_name + str(t)))))
                                # print(t_start_cm, t_spending, datetime.now())
                                cropped = image.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                                cropped.save(
                                    'outputs/caps_of_detections/parking/' + str(
                                        class_name) + str(t) + str('.jpg'))


                    #If a vehicle in the region and being immobile for a while, started to move suddenly and stopped again, then timer will continue thanks to tracking
                    #In order to avoid that and give the vehicle another chance, this will help to remove the entry from the cache matrix and hence it will apear as a new timer
                    # else:
                    #    entry_indx = str(cache_matrix).find(class_name + str(t))
                    #    cache_matrix[cache_matrix.index(
                    #        eval(str(cache_matrix)[entry_indx - 31:entry_indx + len(class_name + str(t)) + 1]))] = None

                    #or you can simply use > cache_matrix.remove(str(cache_matrix)[entry_indx - 31:entry_indx + len(class_name + str(t)) + 1]))



            #Avoid buffer overflow
            if len(frame_matrix)>10^7:
                frame_matrix=[]


            # print(cache_matrix)
            # print(viol_matrix)
            # print(frame_matrix)




            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)
            cv2.fillPoly(frame, np.int32([pts]), 255)

            # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id),
                                                                                                    class_name, (
                                                                                                    int(bbox[0]),
                                                                                                    int(bbox[1]),
                                                                                                    int(bbox[2]),
                                                                                                    int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
