# GeoDesc implementation

TensorFlow implementation of GeoDesc for ECCV'18 paper ["GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints"](https://arxiv.org/abs/1807.06294), Zixin Luo, Tianwei Shen, Lei Zhou, Siyu Zhu, Runze Zhang, Yao Yao, Tian Fang and Long Quan.

## Update 08/04/2019

We improve the patch cropping implementation and now it gets 5 times faster.

## Update 14/08/2018

We have provided an example to test the matching performance of GeoDesc (examples/image_matching.py). See usage below. 

## Requirements

Please use Python 2.7, install NumPy, OpenCV and TensorFlow. To run the image matching example, you may also need to compile [opencv_contrib](https://github.com/opencv/opencv_contrib) to get SIFT support.

## Pre-trained model

Pre-trained GeoDesc model (in TensorFlow Protobuf format) can be found [here](http://home.cse.ust.hk/~zluoag/data/geodesc.pb).

Model in NumPy dictionary is available [here](http://home.cse.ust.hk/~zluoag/data/geodesc.npy), which is more handy to be converted to other formats.

## Example scripts

### 1. Extract features of HPatches

An example script is provided to extract features of HPatches. [HPatches](https://github.com/hpatches/hpatches-benchmark) should be ready in its original format.

After download HPatches, you can start to evaluate GeoDesc: 

```bash
git clone https://github.com/lzx551402/geodesc.git
cd geodesc/model
wget http://home.cse.ust.hk/~zluoag/data/geodesc.pb
cd ../examples
python extract_features_of_hpatches.py \
    --hpatches_root=<hpatches_benchmark>/data/hpatches-release \
    --feat_out_path=<hpatches_benchmark>/data/descriptors
```

After the extraction, you can use HPatches benchmarking tools to evaluate GeoDesc (on split 'full' as GeoDesc is not trained on HPatches):

```bash
cd <hpatches_benchmark>/python
python hpatches_eval.py --descr-name='geodesc' \
    --task=verification --task=matching --task=retrieval --split=full
```

And then display the results:
```bash
python hpatches_results.py --descr-name='geodesc' \
    --task=verification --task=matching --task=retrieval --split=full --results-dir=
```

### 2. Test image matching

As described in the paper, the matching pipeline consists of: i) detect keypoints by SIFT detector, ii) crop patches in the scale space, iii) compute features on cropped patches, and iv) match the two images. If you want to achieve the efficiency reported in the paper and use it for large-scale matching tasks, we strongly suggest you implementing the pipeline in C++ with integrated GPU-based SIFT (e.g., [SIFTGPU](https://github.com/pitzer/SiftGPU)) and GPU-based matcher (e.g., [OpenCV GPU matcher](https://docs.opencv.org/3.4/dd/dc5/classcv_1_1cuda_1_1DescriptorMatcher.html)). We have provided here only a prototype for research purposes. 

To get started, simply run:

```bash
cd examples
python image_matching.py --cf_sift
```

The matching results from SIFT (top) and GeoDesc (bottom) will be displayed. Type `python image_matching.py --h` to view more options and test on your own images.

<p><img src="https://github.com/lzx551402/geodesc/blob/master/img/matching_example.jpg" alt="sample" width="70%"></p>

(Image source: Graffiti sequence in [Heinly benchmark](http://cs.unc.edu/~jheinly/binary_descriptors.html))

## Training code

The ground truth patches used to train GeoDesc are under preparation. 

## Benchmark on [HPatches](https://github.com/hpatches/hpatches-benchmark), mAP

<p><img src="https://github.com/lzx551402/geodesc/blob/master/img/hpatches_results.png" alt="sample" width="70%"></p>

## Benchmark on [Heinly benchmark](http://cs.unc.edu/~jheinly/binary_descriptors.html)

<p><img src="https://github.com/lzx551402/geodesc/blob/master/img/heinly_results.png" alt="sample" width="70%"></p>

## Benchmark on [ETH local features benchmark](https://github.com/ahojnnes/local-feature-evaluation)

<p><img src="https://github.com/lzx551402/geodesc/blob/master/img/eth_results.jpg" alt="sample" width="70%"></p>

## Application on 3D reconstructions

<p><img src="https://github.com/lzx551402/geodesc/blob/master/img/3d_reconstructions.jpg" alt="sample" width="70%"></p>
