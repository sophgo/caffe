
#include <vector>
#include "caffe/layers/padding_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PadImageConstNCHW(const int nthreads, const Dtype* const bottom_data, const int num,
    const int channels, const int height, const int width,
    const int padded_height, const int padded_width,
    const int pad_t, const int pad_l, Dtype value, Dtype* const top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int nc = index / padded_width;
        const int pw = index % padded_width;
        const int ph = nc % padded_height;
        nc /= padded_height;
        const int h = ph - pad_t;
        const int w = pw - pad_l;
        top_data[index] = (h < 0 || w < 0 || h >= height || w >= width)
            ? value
            : bottom_data[(nc * height + h) * width + w];
  }
}

template <typename Dtype>
void PaddingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = top[0]->count();
    const int num = bottom[0]->shape(0);
    const int channels = bottom[0]->shape(1);
    const int height = bottom[0]->shape(2);
    const int width = bottom[0]->shape(3);

    const int padded_height = top[0]->shape(2);
    const int padded_width = top[0]->shape(3);

    PadImageConstNCHW<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count,
        bottom_data,
        num,
        channels,
        height,
        width,
        padded_height,
        padded_width,
        pad_t_,
        pad_l_,
        pad_value_,
        top_data);
}

template <typename Dtype>
void PaddingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    /* Not support */
}
INSTANTIATE_LAYER_GPU_FUNCS(PaddingLayer);
}



