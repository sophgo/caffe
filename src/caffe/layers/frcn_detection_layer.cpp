#include "caffe/layers/frcn_detection_layer.hpp"

namespace caffe {

static void bbox_transform_inv(const float* boxes, const float* deltas, float* pred, int num, int class_num)
{
  for (int i = 0; i < num; ++i) {
    float height = boxes[i*4+3] - boxes[i*4+1] + 1;
    float width = boxes[i*4+2] - boxes[i*4+0] + 1;
    float ctr_x = boxes[i*4+0] + width * 0.5;
    float ctr_y = boxes[i*4+1] + height * 0.5;

    for (int j = 0; j < class_num; ++j) {
      float dx = deltas[i*class_num*4 + j*4 + 0];
      float dy = deltas[i*class_num*4 + j*4 + 1];
      float dw = deltas[i*class_num*4 + j*4 + 2];
      float dh = deltas[i*class_num*4 + j*4 + 3];

      float pred_ctr_x = dx * width + ctr_x;
      float pred_ctr_y = dy * height + ctr_y;
      float pred_w = std::exp(dw) * width;
      float pred_h = std::exp(dh) * height;

      pred[i*class_num*4 + j*4 + 0] = pred_ctr_x - pred_w / 2;
      pred[i*class_num*4 + j*4 + 1] = pred_ctr_y - pred_h / 2;
      pred[i*class_num*4 + j*4 + 2] = pred_ctr_x + pred_w / 2; 
      pred[i*class_num*4 + j*4 + 3] = pred_ctr_y + pred_h / 2;
    }
  }
}

static void nms(detection *dets, int num, float nms_threshold) 
{
  for (int i = 0; i < num; i++) {
    if (dets[i].score == 0) {
      // erased already
      continue;
    }

    float s1 = (dets[i].bbox.x2 - dets[i].bbox.x1 + 1) * (dets[i].bbox.y2 - dets[i].bbox.y1 + 1);
    for (int j = i + 1; j < num; j++) {
      if (dets[j].score == 0) {
        // erased already
        continue;
      }
      if (dets[i].cls != dets[j].cls) {
        // not the same class
        continue;
      }

      float s2 = (dets[j].bbox.x2 - dets[j].bbox.x1 + 1) * (dets[j].bbox.y2 - dets[j].bbox.y1 + 1);

      float x1 = std::max(dets[i].bbox.x1, dets[i].bbox.x1);
      float y1 = std::max(dets[i].bbox.y1, dets[i].bbox.y1);
      float x2 = std::min(dets[i].bbox.x2, dets[i].bbox.x2);
      float y2 = std::min(dets[i].bbox.y2, dets[i].bbox.y2);

      float width = x2 - x1;
      float height = y2 - y1;
      if (width > 0 && height > 0) {
        float iou = width * height / (s1 + s2 - width * height);
        assert(iou <= 1.0f);
        if (iou > nms_threshold) {
          // overlapped, select one to erase
          if (dets[i].score < dets[j].score) {
            dets[i].score = 0;
          } else {
            dets[j].score = 0;
          }
        }
      }
    }
  }
}

template <typename Dtype>
void FrcnDetectionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top)
{
  nms_threshold_ = this->layer_param_.frcn_detection_param().nms_threshold();
  obj_threshold_ = this->layer_param_.frcn_detection_param().obj_threshold();
  keep_topk_ = this->layer_param_.frcn_detection_param().keep_topk();
  class_num_ = this->layer_param_.frcn_detection_param().class_num();
  //std::cout << "nms_threshold: " << nms_threshold_ << ", obj_threshold: " << obj_threshold_
  //          << ", keep_topk_: " << keep_topk_ << ", class_num: " << class_num_ << "\n";
}

template <typename Dtype>
void FrcnDetectionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top)
{
  // [x1, y1, x2, y2, cls, confidence]
  top[0]->Reshape(1, 1, keep_topk_, 6);
}

template <typename Dtype>
void FrcnDetectionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top)
{
  auto top_data = top[0]->mutable_cpu_data();
  // [num, 84]
  const Dtype* bbox_deltas = bottom[0]->cpu_data();
  // [num, 21]
  const Dtype* scores = bottom[1]->cpu_data();
  // [num, 5]
  const Dtype* rois = bottom[2]->cpu_data();

  int num = bottom[0]->shape(0);
  
  //printf("frcn output forward num: %d\n", num);

  std::vector<float> boxes(num * 4, 0);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < 4; ++j) {
      boxes[i*4 + j] = rois[i*5 + j + 1];
      //printf("%f,", boxes[i*4+j]);
    }
    //printf("\n");
  }

  std::vector<float> pred(num * class_num_ * 4, 0);
  float *pred_data = pred.data();
  std::vector<float> deltas(bbox_deltas, bbox_deltas+bottom[0]->count());
  bbox_transform_inv(boxes.data(), deltas.data(), pred_data, num, class_num_);

  int det_num = 0;
  detection dets[num];

  for (int i = 0; i < num; ++i) {
    for (int j = 1; j < class_num_; ++j) {
      if (scores[i*class_num_ + j] > obj_threshold_) {
        dets[det_num].bbox.x1 = pred[i*class_num_*4 + j*4 + 0];
        dets[det_num].bbox.y1 = pred[i*class_num_*4 + j*4 + 1];
        dets[det_num].bbox.x2 = pred[i*class_num_*4 + j*4 + 2];
        dets[det_num].bbox.y2 = pred[i*class_num_*4 + j*4 + 3];
        dets[det_num].cls = j;
        dets[det_num].score = scores[i*class_num_ + j];
        det_num++;
      }
    }
  }

  nms(dets, det_num, nms_threshold_);
  detection dets_nms[det_num];
  int det_idx = 0;
  for (int i = 0; i < det_num; i++) {
    if (dets[i].score > 0) {
      dets_nms[det_idx] = dets[i];
      det_idx ++;
    }
  }

  if (keep_topk_ > det_idx)
      keep_topk_ = det_idx;

  long long count = 0;
  for(int i = 0; i < keep_topk_; ++i) {
    top_data[count++] = dets_nms[i].bbox.x1;
    top_data[count++] = dets_nms[i].bbox.y1;
    top_data[count++] = dets_nms[i].bbox.x2;
    top_data[count++] = dets_nms[i].bbox.y2;
    top_data[count++] = dets_nms[i].cls;
    top_data[count++] = dets_nms[i].score;
  }
}

template <typename Dtype>
void FrcnDetectionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down,
                                              const vector<Blob<Dtype>*>& bottom)
{

}

#ifdef CPU_ONLY
STUB_GPU(FrcnDetectionLayer);
#endif

INSTANTIATE_CLASS(FrcnDetectionLayer);
REGISTER_LAYER_CLASS(FrcnDetection);
}  // namespace caffe