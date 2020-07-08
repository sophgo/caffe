#ifndef CAFFE_YOLO_DETECTION_LAYER_HPP_
#define CAFFE_YOLO_DETECTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

#define MAX_DET 200
#define MAX_DET_RAW 500

typedef struct box_ {
    float x, y, w, h;
} box;

typedef struct detection_ {
    box bbox;
    int cls;
    float score;
} detection;

template <typename Dtype>
class YoloDetectionLayer : public Layer<Dtype> {
public:
  explicit YoloDetectionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "YoloDetection"; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
  float nms_threshold_;
  float obj_threshold_;
  int net_input_h_;
  int net_input_w_;
  int keep_topk_;
  bool tiny_;
};

}  // namespace caffe

#endif  // CAFFE_YOLO_DETECTION_LAYER_HPP_
