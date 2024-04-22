import numpy as np
from numpy import random

import mmcv
from mmdet3d.core.bbox import box_np_ops, CameraInstance3DBoxes
from mmdet3d.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import RandomFlip, Resize, Pad


@PIPELINES.register_module()
class RandomFlipMonoCon(RandomFlip):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 sync_2d=True,
                 flip_ratio_bev_horizontal=0.0,
                 flip_ratio_bev_vertical=None,
                 **kwargs):
        super(RandomFlipMonoCon, self).__init__(
            flip_ratio=flip_ratio_bev_horizontal, **kwargs)
        self.sync_2d = sync_2d
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        assert flip_ratio_bev_vertical is None, 'bev_vertical_flip is not supported'

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Default: horizontal.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are \
                updated in the result dict.
        """
        assert direction in ['horizontal']

        w = input_dict['img_shape'][1]
        cam = np.array(input_dict['cam2img'])
        cam[0, 2] = w - cam[0, 2]
        cam[0, 3] = - cam[0, 3]
        input_dict['cam2img'] = cam

        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))
        for key in input_dict['bbox3d_fields']:
            if 'points' in input_dict:
                input_dict['points'] = input_dict[key].flip(
                    direction, points=input_dict['points'])
            else:
                input_dict[key].flip(direction)

        if 'centers2d' in input_dict:
            assert self.sync_2d is True and direction == 'horizontal', \
                'Only support sync_2d=True and horizontal flip with images'
            w = input_dict['img_shape'][1]
            input_dict['centers2d'][..., 0] = w - input_dict['centers2d'][..., 0]

        if 'gt_kpts_2d' in input_dict:
            w = input_dict['img_shape'][1]

            # flip kpts and adjust kpts order
            gt_kpts_2d = input_dict['gt_kpts_2d'].copy()
            gt_kpts_valid_mask = input_dict['gt_kpts_valid_mask'].copy()
            if len(gt_kpts_2d) > 0:
                gt_kpts_2d[..., 0::2] = w - gt_kpts_2d[..., 0::2]
                num_box, _ = gt_kpts_2d.shape
                gt_kpts_2d = gt_kpts_2d.reshape(num_box, -1, 2)
                gt_kpts_2d[:, [0, 1, 2, 3, 4, 5, 6, 7]] = gt_kpts_2d[:, [1, 0, 3, 2, 5, 4, 7, 6]]
                gt_kpts_valid_mask[:, [0, 1, 2, 3, 4, 5, 6, 7]] \
                    = gt_kpts_valid_mask[:, [1, 0, 3, 2, 5, 4, 7, 6]]
                input_dict['gt_kpts_2d'] = gt_kpts_2d.reshape(num_box, -1)
                input_dict['gt_kpts_valid_mask'] = gt_kpts_valid_mask

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and \
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction', \
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added \
                into result dict.
        """
        # filp 2D image and its annotations
        super(RandomFlipMonoCon, self).__call__(input_dict)

        if self.sync_2d:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio else False
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(sync_2d={self.sync_2d},'
        repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        return repr_str


@PIPELINES.register_module()
class RandomShiftMonoCon(object):
    """Shift the image and box given shift pixels and probability.

    Args:
        shift_ratio (float): Probability of shifts. Default 0.5.
        max_shift_px (int): The max pixels for shifting. Default 32.
        filter_thr_px (int): The width and height threshold for filtering.
            The bbox and the rest of the targets below the width and
            height threshold will be filtered. Default 1.
    """

    def __init__(self, sync_2d=True, shift_ratio=0.5, max_shift_px=32, filter_thr_px=1):
        assert 0 <= shift_ratio <= 1
        assert max_shift_px >= 0
        self.sync_2d = sync_2d
        self.shift_ratio = shift_ratio
        self.max_shift_px = max_shift_px
        self.filter_thr_px = int(filter_thr_px)
        # The key correspondence from bboxes to labels.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }

    def __call__(self, results):
        """Call function to random shift images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Shift results.
        """
        if random.random() < self.shift_ratio:
            img_shape = results['img'].shape[:2]

            random_shift_x = random.randint(-self.max_shift_px,
                                            self.max_shift_px)
            random_shift_y = random.randint(-self.max_shift_px,
                                            self.max_shift_px)
            new_x = max(0, random_shift_x)
            orig_x = max(0, -random_shift_x)
            new_y = max(0, random_shift_y)
            orig_y = max(0, -random_shift_y)

            for key in results.get('bbox_fields', []):
                bboxes = results[key].copy()
                bboxes[..., 0::2] += random_shift_x
                bboxes[..., 1::2] += random_shift_y

                # clip border
                bboxes[..., 0::2] = np.clip(bboxes[..., 0::2], 0, img_shape[1])
                bboxes[..., 1::2] = np.clip(bboxes[..., 1::2], 0, img_shape[0])

                # remove invalid bboxes
                bbox_w = bboxes[..., 2] - bboxes[..., 0]
                bbox_h = bboxes[..., 3] - bboxes[..., 1]
                valid_inds = (bbox_w > self.filter_thr_px) & (
                    bbox_h > self.filter_thr_px)
                # If the shift does not contain any gt-bbox area, skip this
                # image.
                if key == 'gt_bboxes' and not valid_inds.any():
                    return results
                bboxes = bboxes[valid_inds]
                results[key] = bboxes

                # label fields. e.g. gt_labels and gt_labels_ignore
                label_key = self.bbox2label.get(key)
                if label_key in results:
                    results[label_key] = results[label_key][valid_inds]

                if key == 'gt_bboxes':
                    cam = np.array(results['cam2img'])
                    cam[0, 2] += random_shift_x
                    cam[1, 2] += random_shift_y
                    results['cam2img'] = cam

                    if 'gt_labels_3d' in results:
                        results['gt_labels_3d'] = results['gt_labels_3d'][valid_inds]

                    if 'gt_bboxes_3d' in results:
                        box_3d_tensor = results['gt_bboxes_3d'].tensor
                        assert box_3d_tensor.shape[-1] == 7
                        box_3d_tensor = box_3d_tensor[valid_inds]
                        box_3d_tensor = CameraInstance3DBoxes(box_3d_tensor)
                        results['gt_bboxes_3d'] = box_3d_tensor

                    if 'gt_bboxes_3d_cam2' in results:
                        box_3d_tensor = results['gt_bboxes_3d_cam2'].tensor
                        assert box_3d_tensor.shape[-1] == 7
                        box_3d_tensor = box_3d_tensor[valid_inds]
                        box_3d_tensor = CameraInstance3DBoxes(box_3d_tensor)
                        results['gt_bboxes_3d_cam2'] = box_3d_tensor

                    if 'centers2d' in results:
                        centers2d = results['centers2d'].copy()

                        centers2d[..., 0] += random_shift_x
                        centers2d[..., 1] += random_shift_y

                        centers2d = centers2d[valid_inds]
                        results['centers2d'] = centers2d

                    if 'depths' in results:
                        results['depths'] = results['depths'][valid_inds]

                    if 'gt_kpts_2d' in results:
                        gt_kpts_2d = results['gt_kpts_2d'].copy()
                        gt_kpts_2d[..., 0::2] += random_shift_x
                        gt_kpts_2d[..., 1::2] += random_shift_y

                        gt_kpts_2d = gt_kpts_2d[valid_inds]
                        results['gt_kpts_2d'] = gt_kpts_2d
                        results['gt_kpts_valid_mask'] = results['gt_kpts_valid_mask'][valid_inds]

            for key in results.get('img_fields', ['img']):
                img = results[key]
                new_img = np.zeros_like(img)
                img_h, img_w = img.shape[:2]
                new_h = img_h - np.abs(random_shift_y)
                new_w = img_w - np.abs(random_shift_x)
                new_img[new_y:new_y + new_h, new_x:new_x + new_w] \
                    = img[orig_y:orig_y + new_h, orig_x:orig_x + new_w]
                results[key] = new_img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_shift_px={self.max_shift_px}, '
        return repr_str


@PIPELINES.register_module()
class CustomM3DPad(Pad):
    def __call__(self, results):
        super(CustomM3DPad, self).__call__(results)
        return results