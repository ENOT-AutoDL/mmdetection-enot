config_path=configs/ssd/pascal_voc/ssd_mnv2_tune_on_resolution.py
chkpt_path=./workdir/tune_searched_model/epoch_39.pth
out_path=searched_mnv2_ssd.onnx

CUDA_VISIBLE_DEVICES=0 python mmdet/enot_tools/pytorch2onnx_enot.py \
    $config_path \
    $chkpt_path \
    --output-file $out_path\
    --input-img demo/demo.jpg\
    --shape 224\
