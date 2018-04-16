#!/usr/bin/env python

from __future__ import print_function

import cv2 as cv
from cv_bridge import CvBridge
import numpy as np
import rospy

from sensor_msgs.msg import Image
from ros_dnn.msgs import Prediction, Predictions

NAME = 'ros_dnn_detect'


class DNN(object):
    def __init__(
            self, model_path, config_path,
            framework=None,
            height=None,
            width=None):

        self.net = cv.dnn.readNet(model_path, config_path, framework)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        self.height = height
        self.width = width

    def __postprocess(self, frame, predictions):
        layerNames = self.net.getLayerNames()
        lastLayerId = self.net.getLayerId(layerNames[-1])
        firstLayer = self.net.getLayer(0)
        lastLayer = self.net.getLayer(lastLayerId)

        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        confThreshold = 0.5

        if firstLayer.outputNameToIndex('im_info') != -1:
            # Faster-RCNN or R-FCN
            # Network produces output blob with a shape 1x1xNx7 where N is a
            # number of detections and an every detection is a vector of values
            # [batchId, classId, confidence, left, top, right, bottom]
            for detection in predictions[0, 0]:
                confidence = detection[2]
                if confidence > confThreshold:
                    left = int(detection[3])
                    top = int(detection[4])
                    right = int(detection[5])
                    bottom = int(detection[6])
                    classId = int(detection[1]) - 1  # Skip background label
                    drawPred(classId, confidence, left, top, right, bottom)
        elif lastLayer.type == 'DetectionOutput':
            # Network produces output blob with a shape 1x1xNx7 where N is a
            # number of detections and an every detection is a vector of values
            # [batchId, classId, confidence, left, top, right, bottom]
            for detection in predictions[0, 0]:
                confidence = detection[2]
                if confidence > confThreshold:
                    left = int(detection[3] * frameWidth)
                    top = int(detection[4] * frameHeight)
                    right = int(detection[5] * frameWidth)
                    bottom = int(detection[6] * frameHeight)
                    classId = int(detection[1]) - 1  # Skip background label
                    drawPred(classId, confidence, left, top, right, bottom)
        elif lastLayer.type == 'Region':
            # Network produces output blob with a shape NxC where N is a number of
            # detected objects and C is a number of classes + 4 where the first 4
            # numbers are [center_x, center_y, width, height]
            for detection in predictions:
                confidences = detection[5:]
                classId = np.argmax(confidences)
                confidence = confidences[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = center_x - width / 2
                    top = center_y - height / 2
                    drawPred(classId, confidence, left, top, left + width, top + height)

    def predict(self, frame):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Create 4D blob from frame.
        width = self.width if self.width else frameWidth
        height = self.height if self.height else frameHeight
        blob = cv.dnn.blobFromImage(frame, 1.0 (width, height), crop=False)

        # Run model
        self.net.setInput(blob)

        firstLayer = self.net.getLayer(0)
        if firstLayer.outputNameToIndex('im_info') != -1:
            # Faster-RCNN or R-FCN
            frame = cv.resize(frame, (width, height))
            self.net.setInput(
                    np.array([height, width, 1.6], dtype=np.float32),
                    'im_info')
        predictions = self.net.forward()

        return self._postprocess(frame, predictions)


class DetectHandler(object):
    def __init__(self):

        camera_params = rospy.get_param('subscribers/camera')
        rospy.Subscriber(
                *camera_params,
                data_class=Image,
                callback=self.camera_cb)

        predictions_params = rospy.get_params('publishers/predictions')
        rospy.Publisher(
                *predictions_params,
                data_class=Predictions)

        detection_image_params = rospy.get_params('publishers/detection_image')
        rospy.Publisher(
                *detection_image_params,
                data_class=Image)

        model_path = rospy.get_param('model_path')
        config_path = rospy.get_param('config_path')
        framework = rospy.get_param('framework')

        self.dnn = DNN(model_path, config_path, framework)
        self.br = CvBridge()

        rospy.spin()

    def camera_cb(self, data):
        frame = self.br.imgmsg_to_cv2(data)
        predictions = self.dnn.detect(frame)

        # TODO publish results
