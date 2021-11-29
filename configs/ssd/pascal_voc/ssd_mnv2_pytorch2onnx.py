path_to_model = 'path_to_model/model.pth'
work_dir = 'work_dir/pytorch2onnx/'
output_file = work_dir + '/model.onnx'
searched_arch = None
resolution = (300, 300)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)

