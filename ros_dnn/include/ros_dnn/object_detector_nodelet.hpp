/* ROS */
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <image_transport/image_transport.h>
#include <dynamic_reconfigure/server.h>
#include <ros_dnn/ObjectDetectorConfig.h>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/Image.h>
#include <ros_dnn_msgs/Prediction.h>
#include <ros_dnn_msgs/Predictions.h>
#include <ros_dnn_msgs/DetectAction.h>

/* OpenCV */
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace ros_dnn {
    class Prediction {
        public:
            Prediction(int class_id, int confidence, cv::Point x, cv::Point y)
                : class_id(class_id),
                  confidence(confidence),
                  x(x),
                  y(y)
            {
            }
        private:
            int class_id;
            int confidence;
            cv::Point x;
            cv::Point y;
    }

    class ObjectDetectorNodelet: public nodelet::Nodelet {
        public:
            virtual void onInit();

        private:
            ros::NodeHandle nh;
            ros::NodeHandle nh_ns;

            /* Dynamic reconfigure */
            dynamic_reconfigure::Server<ros_dnn::ObjectDetectorConfig> server;
            dynamic_reconfigure::Server<ros_dnn::ObjectDetectorConfig>::CallbackType f;
            void dyn_reconf_cb(ros_dnn::ObjectDetectorConfig &config, uint32_t level);

            /* Neural network */
            cv::dnn::Net net;
            double conf_threshold;
            std::vector<std::string> class_labels;
            int frame_height;
            int frame_width;

            void draw_predictions(int class_id, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
            std::vector<ros_dnn::Prediction> postprocess(cv::Mat& frame, const cv::Mat& out, cv::dnn::Net& net);
            cv::dnn::Net read_network(const std::string& _model, const std::string& _config, const std::string& _framework);
            
            /* Publish/subscribe */
            image_transport::Subscriber sub_img;
            image_transport::Publisher pub_img;
            ros::Publisher pub_pred;

            void camera_cb(const sensor_msgs::ImageConstPtr& msg);
    };
} /* Namespace ros_dnn */
