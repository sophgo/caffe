#ifndef CAFFE_FASTER_RCNN_DETECTION_LAYER_HPP_
#define CAFFE_FASTER_RCNN_DETECTION_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

typedef struct box_ {
    float x1, y1, x2, y2;
} box;

typedef struct detection_ {
    box bbox;
    int cls;
    float score;
} detection;

template <typename Dtype>
class FrcnDetectionLayer : public Layer<Dtype> {
public:
    explicit FrcnDetectionLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const { return "FrcnDetection"; }

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
    int keep_topk_;
    int class_num_;
};

}  // namespace caffe
#endif // CAFFE_FASTER_RCNN_DETECTION_LAYER_HPP_
