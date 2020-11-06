#include <vector>
#include "caffe/layers/padding_layer.hpp"

namespace caffe {

template <typename Dtype>
void PaddingLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  PaddingParameter param = this->layer_param_.padding_param();
  pad_method_ = param.pad_method();
  pad_value_ = param.pad_value();
  pad_t_ = param.pad_t();
  pad_l_ = param.pad_l();
  pad_b_ = param.pad_b();
  pad_r_ = param.pad_r();

  if (pad_method_ != PaddingParameter::CONSTANT)
    LOG(FATAL) << "Only support constant padding so far.";
}

template <typename Dtype>
void PaddingLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> out_shape;
  for (int i = 0; i < bottom[0]->num_axes(); i++) {
    out_shape.push_back(bottom[0]->shape(i));
  }

  out_shape[bottom[0]->num_axes() - 1] += pad_l_ + pad_r_;
  out_shape[bottom[0]->num_axes() - 2] += pad_t_ + pad_b_;
  top[0]->Reshape(out_shape);
}

template <typename Dtype>
void PaddingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int N = bottom[0]->shape(0);
  int channels = bottom[0]->shape(1);
  int height = bottom[0]->shape(2);
  int width = bottom[0]->shape(3);
  int padded_height = top[0]->shape(2);
  int padded_width = top[0]->shape(3);

  const Dtype *input = bottom[0]->cpu_data();
  Dtype *output = top[0]->mutable_cpu_data();

  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < padded_height; ++ph) {
        for (int pw = 0; pw < padded_width; ++pw) {
          int h = ph - pad_t_;
          int w = pw - pad_l_;
          output[ph * padded_width + pw] =
              (h < 0 || w < 0 || h >= height || w >= width)
              ? pad_value_
              : input[h * width + w];
        }
      }
      // Do offset.
      input += height * width;
      output += padded_height * padded_width;
    }
  }

}

template <typename Dtype>
void PaddingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  /* Not support */
}

#ifdef CPU_ONLY
STUB_GPU(PaddingLayer);
#endif

INSTANTIATE_CLASS(PaddingLayer);
REGISTER_LAYER_CLASS(Padding);

}  // namespace caffe

