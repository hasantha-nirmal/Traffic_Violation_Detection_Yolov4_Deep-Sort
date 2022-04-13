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
from skimage.draw import line
import xlwt
from lane_line_extract3 import get_lane_line_co


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
flags.DEFINE_boolean('info', True, 'show detailed info of tracked objects')
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

    # entry_count=0

    # Comapare 2 regions:
    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))

    poly_co_start = []
    poly_co_finish = []

    blanked = np.zeros((1024, 2048), dtype=np.uint8)
    pts_finish = np.array(([1065, 517], [1656, 481], [1808, 504], [1175, 616]))
    # pts_finish = np.array(([261,93], [562,20], [600,25], [301,130]))

    cv2.fillPoly(blanked, np.int32([pts_finish]), 255)
    # print(np.where(blanked == 255))

    x_cord_finish = np.where(blanked == 255)[1]
    y_cord_finish = np.where(blanked == 255)[0]

    for q in range(0, len(x_cord_finish)):
        poly_co_finish.append((x_cord_finish[q], y_cord_finish[q]))

    poly_co_start = get_lane_line_co()[2]

    # Global variables and matrixes
    viol_matrix = []
    final_detect = []


    vid = cv2.VideoCapture('data/video/lane_2.mp4')

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
            # frame = frame[423:998, 806:1408]
            # frame = cv2.resize(frame, (1000,500), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

            # equalize the histogram of the Y channel
            frame[:, :, 0] = cv2.equalizeHist(frame[:, :, 0])

            # convert the YUV image back to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #Marking lane_line co
            for mark_lane in range(0, len(get_lane_line_co()[1])):
                frame[get_lane_line_co()[1][mark_lane], get_lane_line_co()[0][mark_lane]] = [255,0,0]





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
        #allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['car','motorbike','truck','bus','bicycle']

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

            # centre_cord_x = int((int(bbox[0]) + int(bbox[2])) / 2)
            # centre_cord_y = int((int(bbox[1]) + int(bbox[3])) / 2)
            #
            # x_min = int(centre_cord_x - (int(bbox[2])-int(bbox[0]))/4)-15
            # y_min = int(centre_cord_y - (int(bbox[3])-int(bbox[1]))/4)-15
            # x_max = int(centre_cord_x + (int(bbox[2])-int(bbox[0]))/4)+15
            # y_max = int(centre_cord_y + (int(bbox[3])-int(bbox[1]))/4)+15


            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            #cv2.rectangle(frame, (centre_cord_x - 30, centre_cord_y - 30), (centre_cord_x + 30, centre_cord_y + 30), color,3)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]) , int(bbox[3])), color, 2)
            #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                        (255, 0, 0), 2)

            # Coordinates of rectangle  boxes
            cv2.fillPoly(frame, np.int32([pts_finish]), 255)
            #cv2.fillPoly(frame, np.int32([pts_start]), 255)



            # # Define the speed detection fns
            # b_box_co = []
            #
            # # Change accordingly
            # X, Y = np.mgrid[centre_cord_x - 30:centre_cord_x + 30, centre_cord_y - 30:centre_cord_y + 30]
            # for m in range(0, len(X)):
            #     for n in range(0, len(X[0])):
            #         b_box_co.append((X[m][n], Y[m][n]))

            bbox_bottom_line_co = list(
                zip(*line(*(int(bbox[0])-40, int(bbox[3])), *(int(bbox[2]) , int(bbox[3])))))







            if len(intersection(poly_co_start, bbox_bottom_line_co)) > 0 and (str((class_name+str(t))) not in str(viol_matrix)) and (str((class_name+str(t))) != 'truck1') :
                t_start = datetime.now()
                #t_start_global=t_start
                viol_matrix.append((str(t_start), class_name+str(t)))
                cropped_at_spot = image.crop((int(bbox[0])-100, int(bbox[1])-100, int(bbox[2]) + 100, int(bbox[3]) + 100))
                cropped_at_spot.save(
                    'outputs/on_the_spot/' + str(
                        class_name) + str(t) + str('_') + str(frame_num) + str('.jpg'))

                #print(t_start,class_name,str(t))
                print(viol_matrix)

            if len(intersection(poly_co_finish, bbox_bottom_line_co)) > 0 and (str((class_name+str(t))) in str(viol_matrix)) and (str((class_name+str(t))) not in str(final_detect)) and (str((class_name+str(t))) != 'truck1') :


                #Extract start time from viol matrix
                index=str(viol_matrix).find(str((class_name+ str(t))))
                t_start_cm=str(viol_matrix)[index-28:index-11]

                sheet.write(row_num, 0, str(t_start_cm), style)
                sheet.write(row_num, 1, str(class_name) + str(t), style)
                row_num += 1
                workbook.save('outputs/xlsx/lane_line/details.xls')
                # entry_count+=1
                final_detect.append((class_name + str(t)))
                print(final_detect)
                cropped = image.crop((int(bbox[0]), int(bbox[1]), int(bbox[2])+100, int(bbox[3])+100))
                cropped.save('outputs/caps_of_detections/lane_line/' + str(
                    class_name) + str(t) + str('.jpg'))





            if (str((class_name+ str(t))) in str(final_detect)) and (str((class_name+str(t))) != 'truck1'):
                cropped = image.crop((int(bbox[0]), int(bbox[1]), int(bbox[2])+100, int(bbox[3])+100))
                if not os.path.isdir('outputs/caps_of_detections_all/lane_line/'  + str(class_name) + str(t)):
                    os.makedirs('outputs/caps_of_detections_all/lane_line/'+ str(class_name) + str(t))
                    cropped.save(
                        'outputs/caps_of_detections_all/lane_line/' + str(class_name) + str(t) +str('/')+str(frame_num)+ str('.jpg'))

                if os.path.isdir(
                    'outputs/caps_of_detections_all/lane_line/' + str(class_name) + str(t)):
                    cropped.save(
                        'outputs/caps_of_detections_all/lane_line/' + str(
                            class_name) + str(t) + str('/') + str(frame_num) + str('.jpg'))


























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
