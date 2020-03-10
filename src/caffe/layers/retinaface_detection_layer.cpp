#include <algorithm>
#include <vector>

#include "caffe/layers/retinaface_detection_layer.hpp"

namespace caffe {


template <typename Dtype>
void RetinaFaceDetectionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{
    nms_threshold_ = this->layer_param_.retinaface_detection_param().nms_threshold();
    confidence_threshold_ = this->layer_param_.retinaface_detection_param().confidence_threshold();
    keep_topk_ = this->layer_param_.retinaface_detection_param().keep_topk();
}

template <typename Dtype>
void RetinaFaceDetectionLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{
    top[0]->Reshape(1, 1, keep_topk_, 15);
}

template <typename Dtype>
void RetinaFaceDetectionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top)
{
    auto top_data = top[0]->mutable_cpu_data();

    std::vector<FaceInfo> infos;
    size_t bottom_size = bottom.size();
    assert(bottom_size == 9);
   // auto face_rpn_cls_prob_reshape_stride8 = bottom[0]->cpu_data();
   // auto face_rpn_bbox_pred_stride8 = bottom[1]->cpu_data();
   // auto face_rpn_landmark_pred_stirde8 = bottom[2]->cpu_data();
   // auto face_rpn_cls_prob_reshape_stride16 = bottom[3]->cpu_data();
   // auto face_rpn_bbox_pred_stride16 = bottom[4]->cpu_data();
   // auto face_rpn_landmark_pred_stride16 = bottom[5]->cpu_data();
   // auto face_rpn_cls_prob_reshape_stride32 = bottom[6]->cpu_data();
   // auto face_rpn_bbox_pred_stride32 = bottom[7]->cpu_data();
   // auto face_rpn_landmark_pred_stride32 = bottom[8]->cpu_data();

    for (size_t i = 0; i < feature_stride_fpn_.size(); ++i) {
        int stride = feature_stride_fpn_[i];
        auto landmark_data = bottom[bottom_size-3*i-1]->cpu_data();
        size_t landmark_count = bottom[bottom_size-3*i-1]->count();

        auto bbox_data = bottom[bottom_size-3*i-2]->cpu_data();
        size_t bbox_count = bottom[bottom_size-3*i-2]->count();

        auto score_data = bottom[bottom_size-3*i-3]->cpu_data();
        size_t score_count = bottom[bottom_size-3*i-3]->count();

        size_t height = bottom[bottom_size-3*i-3]->height();
        size_t width = bottom[bottom_size-3*i-3]->width();

        std::vector<float> score(score_data + score_count / 2, score_data + score_count);
        std::vector<float> bbox(bbox_data, bbox_data + bbox_count);
        std::vector<float> landmark(landmark_data, landmark_data + landmark_count);

        int count = height * width;
        std::string key = "stride" + std::to_string(stride);
        auto anchors_fpn = anchors_fpn_[key];
        auto num_anchors = num_anchors_[key];

        std::vector<AnchorBox> anchors = anchors_plane(height, width, stride, anchors_fpn);

        for(size_t num = 0; num < num_anchors; ++num) {
            for(size_t j = 0; j < count; ++j) {
                float confidence = score[j+count*num];
                if (confidence < confidence_threshold_)
                    continue;

                float dx = bbox[j+count*(0+num*4)];
                float dy = bbox[j+count*(1+num*4)];
                float dw = bbox[j+count*(2+num*4)];
                float dh = bbox[j+count*(3+num*4)];
                std::vector<float> bbox_deltas{dx,dy,dw,dh};
                auto bbox = bbox_pred(anchors[j+count*num], bbox_deltas);

                std::vector<float> landmark_deltas(10,0);
                for(size_t k = 0; k < 5; ++k) {
                    landmark_deltas[k] = landmark[j+count*(num*10+k*2)];
                    landmark_deltas[k+5] = landmark[j+count*(num*10+k*2+1)];
                }

                auto landmark = landmark_pred(anchors[j+count*num], landmark_deltas);

                FaceInfo info;
                info.x1 = bbox[0];
                info.y1 = bbox[1];
                info.x2 = bbox[2];
                info.y2 = bbox[3];
                info.score = confidence;
                for(int idx = 0; idx < 5; ++idx) {
                    info.x[i] = landmark[i];
                    info.y[i] = landmark[i+5];
                }

                infos.push_back(info);
            }
        }
    }

    auto preds = nms(infos, nms_threshold_);

    long long count = 0;
    for(int i = 0; i < keep_topk_; ++i) {
        top_data[count++] = preds[i].x1;
        top_data[count++] = preds[i].y1;
        top_data[count++] = preds[i].x2;
        top_data[count++] = preds[i].y2;
        top_data[count++] = preds[i].score;
        for(int j = 0; j < 5; ++j) {
            top_data[count++] = preds[i].x[j];
            top_data[count++] = preds[i].y[j];
        }

        std::cout << "x1= " << preds[i].x1 << ",y1= " << preds[i].y1
            << ",x2= " << preds[i].x2 << ",y2= " << preds[i].y2 << std::endl;
    }
}

template <typename Dtype>
void RetinaFaceDetectionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down,
                                              const vector<Blob<Dtype>*>& bottom)
{

}


#ifdef CPU_ONLY
STUB_GPU(RetinaFaceDetectionLayer);
#endif

INSTANTIATE_CLASS(RetinaFaceDetectionLayer);
REGISTER_LAYER_CLASS(RetinaFaceDetection);
}  // namespace caffe
