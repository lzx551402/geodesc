# GeoDesc implementation

TensorFlow implementation of GeoDesc for ECCV'18 paper ["GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints"](https://arxiv.org/abs/1807.06294), Zixin Luo, Tianwei Shen, Lei Zhou, Siyu Zhu, Runze Zhang, Yao Yao, Tian Fang and Long Quan.

## Requirements

Please use Python 2.7, install NumPy, OpenCV and TensorFlow. 

## Pre-trained model

Pre-trained GeoDesc model (in TensorFlow Protobuf format) can be found [here](http://home.cse.ust.hk/~zluoag/data/geodesc.pb).

Model in NumPy dictionary is available [here](http://home.cse.ust.hk/~zluoag/data/geodesc.npy), which is more handy to be parsed to other formats.

## Example script

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
