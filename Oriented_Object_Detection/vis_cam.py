# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path
from functools import partial

import cv2
import mmcv
import numpy as np
from mmcv import Config, DictAction

from mmrotate.utils.det_cam_visualizer import (DetAblationLayer,
                                            DetBoxScoreTarget, DetCAMModel,
                                            DetCAMVisualizer, EigenCAM,
                                            FeatmapAM, reshape_transform)

try:
    from pytorch_grad_cam import (AblationCAM, EigenGradCAM, GradCAM,
                                  GradCAMPlusPlus, LayerCAM, XGradCAM)
    from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image

except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install '
                      '3rd party package pytorch_grad_cam.')

GRAD_FREE_METHOD_MAP = {
    'ablationcam': AblationCAM,
    'eigencam': EigenCAM,
    # 'scorecam': ScoreCAM, # consumes too much memory
    'featmapam': FeatmapAM
}

GRAD_BASE_METHOD_MAP = {
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'xgradcam': XGradCAM,
    'eigengradcam': EigenGradCAM,
    'layercam': LayerCAM
}

ALL_METHODS = list(GRAD_FREE_METHOD_MAP.keys() | GRAD_BASE_METHOD_MAP.keys())


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')

    parser.add_argument('--img', default='dataset/hole/train/images/522.png')
    parser.add_argument('--config', default='configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90_original.py')
    parser.add_argument('--checkpoint', default='weights/resnet34_epoch_20_my.pth')

    parser.add_argument('--out-file', default='prediction_demo.png', help='Path to output file')
    parser.add_argument('--cam-file', default='cam_demo.png', help='Path to cam file')

    parser.add_argument('--device', default='cuda:0', help='cuda:0 or cpu')
    parser.add_argument('--no-norm-in-bbox', default=False, help='Norm in bbox of cam image')
    parser.add_argument(
        '--method',
        default='layercam',  # eigencam, featmapam, gradcam++(only for single layer)
        help='Type of method to use, supports '
        f'{", ".join(ALL_METHODS)}.')

    parser.add_argument(
        '--target-layers',
        default=['neck.fpn_convs[2]'],
        nargs='+',
        type=str,
        help='The target layers to get CAM, if not set, the tool will '
        'specify the backbone')


    parser.add_argument(
        '--preview-model',
        default=False,
        action='store_true',
        help='To preview all the model layers')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--topk',
        type=int,
        default=10,
        help='Topk of the predicted result to visualizer')
    parser.add_argument(
        '--max-shape',
        nargs='+',
        type=int,
        default=20,
        help='max shapes. Its purpose is to save GPU memory. '
        'The activation map is scaled and then evaluated. '
        'If set to -1, it means no scaling.')
    parser.add_argument(
        '--aug-smooth',
        default=False,
        action='store_true',
        help='Wether to use test time augmentation, default not to use')
    parser.add_argument(
        '--eigen-smooth',
        default=False,
        action='store_true',
        help='Reduce noise by taking the first principle componenet of '
        '``cam_weights*activations``')
    parser.add_argument('--out-dir', default=None, help='dir to output file')

    # Only used by AblationCAM
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='batch of inference of AblationCAM')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--ratio-channels-to-ablate',
        type=int,
        default=0.5,
        help='Making it much faster of AblationCAM. '
        'The parameter controls how many channels should be ablated')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    if args.method.lower() not in (GRAD_FREE_METHOD_MAP.keys()
                                   | GRAD_BASE_METHOD_MAP.keys()):
        raise ValueError(f'invalid CAM type {args.method},'
                         f' supports {", ".join(ALL_METHODS)}.')

    return args


def init_model_cam(args, cfg, target_layerses):
    model = DetCAMModel(
        cfg, args.checkpoint, args.score_thr, device=args.device)
    if args.preview_model:
        print(model.detector)
        print('\n Please remove `--preview-model` to get the CAM.')
        return

    target_layers = []
    # for target_layer in args.target_layers:
    for target_layer in target_layerses:
        try:
            target_layers.append(eval(f'model.detector.{target_layer}'))
        except Exception as e:
            print(model.detector)
            raise RuntimeError('layer does not exist', e)

    extra_params = {
        'batch_size': args.batch_size,
        'ablation_layer': DetAblationLayer(),
        'ratio_channels_to_ablate': args.ratio_channels_to_ablate
    }

    if args.method in GRAD_BASE_METHOD_MAP:
        method_class = GRAD_BASE_METHOD_MAP[args.method]
        is_need_grad = True
        assert args.no_norm_in_bbox is False, 'If not norm in bbox, the ' \
                                              'visualization result ' \
                                              'may not be reasonable.'
    else:
        method_class = GRAD_FREE_METHOD_MAP[args.method]
        is_need_grad = False

    max_shape = args.max_shape
    if not isinstance(max_shape, list):
        max_shape = [args.max_shape]
    assert len(max_shape) == 1 or len(max_shape) == 2

    det_cam_visualizer = DetCAMVisualizer(
        method_class,
        model,
        target_layers,
        reshape_transform=partial(
            reshape_transform, max_shape=max_shape, is_need_grad=is_need_grad),
        is_need_grad=is_need_grad,
        extra_params=extra_params)
    return model, det_cam_visualizer


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    'test'
    from mmdet.apis import inference_detector, init_detector, show_result_pyplot
    model_test = init_detector(args.config, args.checkpoint, device=args.device)
    print(model_test)
    result = inference_detector(model_test, args.img)
    # show the results
    show_result_pyplot(
        model_test,
        args.img,
        result,
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)

    '******************************  Here  *********************************'
    grayscale_cam_all = np.zeros((1024, 1024))
    target_layers = []
    target_layers.append('neck.fpn_convs[0]')
    target_layers.append('neck.fpn_convs[1]')
    target_layers.append('neck.fpn_convs[2]')
    target_layers.append('neck.fpn_convs[3]')
    for i, target_layer in enumerate(target_layers):
        target_layeres = []
        target_layeres.append(target_layer)
        model, det_cam_visualizer = init_model_cam(args, cfg, target_layeres)

        images = args.img
        if not isinstance(images, list):
            images = [images]

        for image_path in images:
            image = cv2.imread(image_path)
            model.set_input_data(image)
            result = model()[0]

            bboxes = result['bboxes'][..., :5]
            scores = result['bboxes'][..., 5]
            labels = result['labels']
            assert bboxes is not None and len(bboxes) > 0

            # if args.topk > 0:  # Topk of the predicted result to visualizer
            #     idxs = np.argsort(-scores)
            #     bboxes = bboxes[idxs[:args.topk]]
            #     labels = labels[idxs[:args.topk]]

            targets = [
                DetBoxScoreTarget(bboxes=bboxes, labels=labels)
            ]

            if args.method in GRAD_BASE_METHOD_MAP:
                model.set_return_loss(True)
                model.set_input_data(image, bboxes=bboxes, labels=labels)
                det_cam_visualizer.switch_activations_and_grads(model)

            grayscale_cam = det_cam_visualizer(
                image,
                targets=targets,
                aug_smooth=args.aug_smooth,
                eigen_smooth=args.eigen_smooth)

            # grayscale_cam_all += grayscale_cam
            grayscale_cam_all = np.maximum(grayscale_cam_all, grayscale_cam)
            print('')

    cam_all_max, cam_all_min = np.amax(grayscale_cam_all), np.amin(grayscale_cam_all)
    grayscale_cam_all = (grayscale_cam_all - cam_all_min) / (cam_all_max - cam_all_min)
    # grayscale_cam_all = scale_cam_image(grayscale_cam_all)

    # with_norm_in_bboxes 是否归一化到检测框内
    image_with_bounding_boxes = det_cam_visualizer.show_cam(image, bboxes, labels, grayscale_cam_all, draw_boxxes=False, with_norm_in_bboxes=False)

    # i_path = '_' + str(i) + '.png'
    # cam_out_dir = args.cam_file.replace('.png', i_path)
    cam_out_dir = args.cam_file
    if cam_out_dir:
        mmcv.imwrite(image_with_bounding_boxes, cam_out_dir)
    else:
        cv2.namedWindow(os.path.basename(image_path), 0)
        cv2.imshow(os.path.basename(image_path), image_with_bounding_boxes)
        cv2.waitKey(0)

    if args.method in GRAD_BASE_METHOD_MAP:
        model.set_return_loss(False)
        det_cam_visualizer.switch_activations_and_grads(model)


if __name__ == '__main__':
    main()
