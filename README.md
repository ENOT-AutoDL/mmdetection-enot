<div style="overflow:auto;">
    <img src="resources/mmdet-logo.png" alt="mmdet" width="300" />
    <img src="resources/ENOT_logo-green.png" alt="mmdet" width="300" />
</div>

[MMDetection documentation](https://mmdetection.readthedocs.io/en/v2.16.0/)

[ENOT documentation](https://enot-autodl.rtd.enot.ai/en/v2.8.0/)

## Introduction

This is Object Detection and Instance Segmentation toolbox integrated with Neural Architecture Search technology from [ENOT](https://enot.ai/).

Project is based on open-mmlab's [MMDetection](https://github.com/open-mmlab/mmdetection)

![demo image](resources/coco_test_12510.jpg)

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation.

## <div align="center">Getting started with the combination of ENOT and MMDetection</div>

This project was made to simplify Neural Architecture Search of models from MMDetection toolbox. We currently support optimization of
three models: `ssd_mobilenetv2`, `yolov3` and `retinanet`. If you want to use other mmdetection models - please contact the
[ENOT team](https://enot.ai/#rec321431603).

We conducted several experiments with `ssd mobilenetv2`, `yolov3` and `retinanet` models on `pascal VOC` dataset. Upon request, we
can share our experience.

With ENOT framework you can reduce your model latency by sacrificing as small mean average precision as possible. To
achieve this goal, we can help you to select the best architecture and the best resolution for your problem.

To apply ENOT, you should do three additional steps.

1. The first step is named `pretrain`. During this procedure you are training all the models from your search space on 
   your own task. `Pretrain` usually takes ~5-7 times longer than single model training despite training millions of
   models. You can start `pretrain` from your baseline checkpoint.
2. The second step searches the best architecture (and, additionally, can search the best resolution) to fit your data.
3. The third step is tuning of searched model.

To estimate gains from ENOT framework usage, you should fairly compare your baseline model and ENOT-searched model.
Baseline model is the best one you achieved by using MMDetection framework. You should tune the model you found with ENOT
with the same setup you trained your baseline model (to exclude unfair comparison).

`Pretrain` stage tips:
* You should organize `pretrain` procedure to be as close to your baseline training as possible. This means that you
  should use the same hyperparameters (augmentation, weight decay, image size, learning rate, etc). One exception is
  that `pretrain` procedure usually benefit from training with more epochs, but this would require more computation 
  resources, so we suggest keeping the same number of training epochs, too.
* `Pretrain` is a resource-intensive procedure, so you should probably use multi-GPU training to make this procedure
  faster.

`Search` stage tips:
* Copy training hyperparameters from your baseline setup. Set <font color="red">**lr0=0.01, momentum=0.0, weight_decay=0.0**</font>.
* Use Adam or RAdam optimizer.
* `target_latency` should be larger than the minimal model latency in the search space.

## <div align="center">Running ENOT optimization</div>

This section describes how you can run neural architecture search with ENOT for Mobilenet SSD on Pascal VOC.

1. Setting up your baseline
   * To start with ENOT, you need to train your baseline model. Baseline model should show your current performance (
     i.e. mean average precision at certain threshold, Precision, Recall, ...). You should also measure its execution
     time (latency), or use latency proxy such as million multiply-add operations (as used in
     [MobileNetv2](https://arxiv.org/abs/1801.04381) article).
   * For multiply-add calculation you can use `measure_latency.sh`
   * To train baseline model on Pascal VOC dataset - set variable `work_dir` in `configs/ssd/pascal_voc/ssd_mnv2_train.py`
     and run `train_baseline.sh` script.
2. Multiresolution pretrain
   * `Multiresolution pretrain`, is required for `search with resolution`.
     During this phase different operations in architecture trains work with each other on different input resolutions.
   * To run `multiresolution pretrain` set parameters `resolution_range`, `work_dir`, `baseline_ckpt`
     and other hyperparameters in `configs/ssd/pascal_voc/ssd_mnv2_multiresolution_pretrain.py`
   * Launch `multiresolution_pretrain.sh` script to start `multiresolution pretrain` procedure.
3. Single resolution pretrain
   * Simple `pretrain` runs on fixed resolution. The purpose of this phase is to train operations work with each other.
   * Important parameters for `pretrain` in `configs/ssd/pascal_voc/ssd_mnv2_pretrain.py` are:
     `baseline_ckpt` and `work_dir`.
   * Other hyperparameters can be changed according to your task.
   * To run pretrain you can use `pretrain.sh` script.
4. Search with resolution
   * In this case we run NAS for different resolutions,
     and choose architecture and resolution compatible with latency constraint with maximum metric value.
   * Launch `search_with_resolution.sh` to search an optimal architecture for your task 
     (you should specify `target_latency` in `configs/ssd/pascal_voc/ssd_mnv2_search_with_resolution.py` needed
     for your project).
   * Other important parameters for `search with resolution` are: `resolution_tolerance`, `baseline_ckpt`, `work_dir`.
6. Search architecture on fixed resolution
   * If you want to search optimal architecture faster and don't want to change input image resolution you can use `search.sh`
   * Before run the script you should set `target_latency`, `baseline_ckpt`, `work_dir`, variables in `configs/ssd/pascal_voc/ssd_mnv2_search.py`
7. Training the model found with ENOT
   * Get the last reported architecture from the search run (which is stored in list with integers),
     copy it and paste to variable `searched_arch` in `configs/ssd/pascal_voc/ssd_mnv2_tune.py` 
     or `configs/ssd/pascal_voc/ssd_mnv2_tune_on_resolution.py` config.
   * Set variables `work_dir` and `ss_checkpoint`.
   * If you set `ss_checkpoint` and don't set `searched_arch` architecture indices will be extracted from search space checkpoint.
   * Launch `train_enot_model_scratch.sh` script to train found architecture from search space.

## <div align="center">Contact</div>
For issues related to ENOT framework contact [ENOT team](https://enot.ai/#rec321431603).


## Citation

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
