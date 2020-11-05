// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
//
#define USE_OPENCV
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using namespace std;

class Detector {
public:
    Detector(const string& model_file,
             const string& weights_file,
             const string& mean_file,
             const string& mean_value);
    // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax]
    std::vector<vector<float> > Detect(const cv::Mat& img);

private:
    void SetMean(const string& mean_file, const string& mean_value);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels);

private:
    boost::shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
};

Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(weights_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];   // data
    num_channels_ = input_layer->channels();             // channels
    CHECK(num_channels_ == 3 || num_channels_ == 1)
            << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height()); // width & height

    /* Load the binaryproto mean file. */
    SetMean(mean_file, mean_value);
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
    Blob<float>* input_layer = net_->input_blobs()[0];        // data layer
    input_layer->Reshape(1, num_channels_,                    // reshape input from NWHC -> NCHW
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* result_blob = net_->output_blobs()[0];      // output data
    const float* result = result_blob->cpu_data();           // cpu
    const int num_det = result_blob->height();               // num detect
    vector<vector<float> > detections;                       // for saving detect result
    for (int k = 0; k < num_det; ++k) {
        if (result[0] == -1) {
            // Skip invalid detection.
            result += 7;    //why add 7? -> Detection format: [image_id, label, score, xmin, ymin, xmax, ymax]
            continue;
        }
        vector<float> detection(result, result + 7);    //copy [result,result + 7) into vector
        detections.push_back(detection);
        result += 7;
    }
    return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
    cv::Scalar channel_mean;
    if (!mean_file.empty()) {
        CHECK(mean_value.empty()) <<
                                     "Cannot specify mean_file and mean_value at the same time";
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), num_channels_)
                << "Number of channels of mean file doesn't match input layer.";

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<cv::Mat> channels;
        float* data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }

        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        cv::merge(channels, mean);

        /* Compute the global mean pixel value and create a mean image
         * filled with this value. */
        channel_mean = cv::mean(mean);
        mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
    }
    if (!mean_value.empty()) {
        CHECK(mean_file.empty()) <<
                                    "Cannot specify mean_file and mean_value at the same time";
        stringstream ss(mean_value);
        vector<float> values;
        string item;
        while (getline(ss, item, ',')) {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        CHECK(values.size() == 1 || values.size() == num_channels_) <<
                                                                       "Specify either 1 mean_value or as many as channels: " << num_channels_;

        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
                            cv::Scalar(values[i]));
            channels.push_back(channel);
        }
        cv::merge(channels, mean_);
    }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Detector::Preprocess(const cv::Mat& img,
                          std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}

DEFINE_string(mean_file, "",
              "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
              "If specified, can be one value or can be same as image channels"
              " - would subtract from the corresponding channel). Separated by ','."
              "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
              "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "/home/gaokechen/MobileNet-YOLO/data/output.txt",
              "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.7,
              "Only store detections with score higher than the threshold.");

//vector<string> labels = {"background",
//                         "aeroplane", "bicycle","bird", "boat", "bottle",
//                         "bus", "car", "cat","chair","cow",
//                         "diningtable","dog","horse","motorbike","person",
//                         "pottedplant","sheep","sofa","train","tvmonitor"};

vector<string> labels = {"background",
                         "sclothes", "aclothes","clothes", "glasses",
                         "nohat", "hair", "nohair", "helmets","vests"};


int main(int argc, char** argv) {

    const string& model_file = "/home/gaokechen/MobileNet-YOLO/models/yolov3/mobilenet_yolov3_lite_deploy.prototxt";
    const string& weights_file = "/home/gaokechen/MobileNet-YOLO/models/yolov3/ehs_deploy_iter_2000.caffemodel";
    const string& mean_file = FLAGS_mean_file;
    const string& mean_value = FLAGS_mean_value;
    const string& file_type = FLAGS_file_type;
    const string& out_file = FLAGS_out_file;
    const float confidence_threshold = FLAGS_confidence_threshold;

    // Initialize the network.
    Detector detector(model_file, weights_file, mean_file, mean_value);

    // Set the output mode.
    std::streambuf* buf = std::cout.rdbuf();
    std::ofstream outfile;
    if (!out_file.empty()) {
        outfile.open(out_file.c_str());
        if (outfile.good()) {
            buf = outfile.rdbuf();
        }
    }
    std::ostream out(buf);

    // Process image one by one.
    std::ifstream infile("/home/gaokechen/MobileNet-YOLO/data/MyDataset/mytest.txt");
    std::string file;

    while (infile >> file)
    {
        if (file_type == "image")
        {
            cv::Mat img = cv::imread(file, -1);
            CHECK(!img.empty()) << "Unable to decode image " << file;
            std::vector<vector<float> > detections = detector.Detect(img);

            /* Print the detection results. */
            for (int i = 0; i < detections.size(); ++i) {
                const vector<float>& d = detections[i];
                // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
                CHECK_EQ(d.size(), 7);
                const float score = d[2];
                if (score >= confidence_threshold) {
                    //save to file
                    out << file << " ";
                    out << static_cast<int>(d[1]) << " ";
                    out << score << " ";
                    out << static_cast<int>(d[3] * img.cols) << " ";              //xmin
                    out << static_cast<int>(d[4] * img.rows) << " ";              //ymin
                    out << static_cast<int>(d[5] * img.cols) << " ";              //xmax
                    out << static_cast<int>(d[6] * img.rows) << std::endl;        //ymax
                    //plot
                    int x = static_cast<int>(d[3] * img.cols);
                    int y = static_cast<int>(d[4] * img.rows);
                    int width = static_cast<int>(d[5] * img.cols) - x;
                    int height = static_cast<int>(d[6] * img.rows) - y;
                    Rect rect(max(x,0), max(y,0), width, height);
                    rectangle(img, rect, Scalar(0,255,0));
                    string sco = to_string(score).substr(0, 5);
                    putText(img, labels[static_cast<int>(d[1])] + ":" + sco, Point(max(x, 0), max(y + height / 2, 0)),
                       FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0));
                }
            }
            imshow("image_show",img);
            //按任意键切换图片
            waitKey(0);
        } else if (file_type == "video") {

            cv::VideoCapture cap(file);
            if (!cap.isOpened()) {
                LOG(FATAL) << "Failed to open video: " << file;
            }
            cv::Mat img;
            int frame_count = 0;
            while (true) {
                bool success = cap.read(img);
                if (!success) {
                    LOG(INFO) << "Process " << frame_count << " frames from " << file;
                    break;
                }
                CHECK(!img.empty()) << "Error when read frame";
                std::vector<vector<float> > detections = detector.Detect(img);

                /* Print the detection results. */
                for (int i = 0; i < detections.size(); ++i) {
                    const vector<float>& d = detections[i];
                    // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
                    CHECK_EQ(d.size(), 7);
                    const float score = d[2];
                    if (score >= confidence_threshold) {
                        //save to file
                        out << file << "_";
                        out << std::setfill('0') << std::setw(6) << frame_count << " ";
                        out << static_cast<int>(d[1]) << " ";
                        out << score << " ";
                        out << static_cast<int>(d[3] * img.cols) << " ";
                        out << static_cast<int>(d[4] * img.rows) << " ";
                        out << static_cast<int>(d[5] * img.cols) << " ";
                        out << static_cast<int>(d[6] * img.rows) << std::endl;
                        //plot
                        int x = static_cast<int>(d[3] * img.cols);
                        int y = static_cast<int>(d[4] * img.rows);
                        int width = static_cast<int>(d[5] * img.cols) - x;
                        int height = static_cast<int>(d[6] * img.rows) - y;
                        Rect rect(max(x,0), max(y,0), width, height);
                        rectangle(img, rect, Scalar(0,255,0));
                        //string sco = to_string(score).substr(0, 5);
                        //putText(img, labels[static_cast<int>(d[1])] + ":" + sco, Point(max(x, 0), max(y + height / 2, 0)),
                        //   FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0));
                    }
                }
                imshow("video_show",img);
                waitKey(30);
                ++frame_count;
            }
            if (cap.isOpened()) {
                cap.release();
            }
        }else if (file_type == "camera"){
            cv::VideoCapture capture;
            capture.open(0);
            while(capture.isOpened()){
                //frame存储每一帧图像
                Mat img;
                //读取当前帧
                capture >> img;
                std::vector<vector<float> > detections = detector.Detect(img);
                for (int i = 0; i < detections.size(); ++i) {
                    const vector<float>& d = detections[i];
                    // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
                    CHECK_EQ(d.size(), 7);
                    const float score = d[2];
                    if (score >= confidence_threshold) {
                        //plot
                        int x = static_cast<int>(d[3] * img.cols);
                        int y = static_cast<int>(d[4] * img.rows);
                        int width = static_cast<int>(d[5] * img.cols) - x;
                        int height = static_cast<int>(d[6] * img.rows) - y;
                        Rect rect(max(x,0), max(y,0), width, height);
                        rectangle(img, rect, Scalar(0,255,0));
                        //string sco = to_string(score).substr(0, 5);
                        //putText(img, labels[static_cast<int>(d[1])] + ":" + sco, Point(max(x, 0), max(y + height / 2, 0)),
                        //   FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0));
                    }
                }
                //显示当前视频
                imshow("camera", img);
                //延时30ms,按下任何键退出
                if (waitKey(30) >= 0)
                    break;
            }
        }else {
            LOG(FATAL) << "Unknown file_type: " << file_type;
        }
    }
    return 0;
}
#else
int main(int argc, char** argv) {
    LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
