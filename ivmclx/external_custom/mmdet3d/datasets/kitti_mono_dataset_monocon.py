import copy
import numpy as np

from mmdet3d.datasets import DATASETS
from mmdet3d.core.bbox import Box3DMode, CameraInstance3DBoxes
from mmdet3d.datasets.kitti_mono_dataset import KittiMonoDataset

EPS = 1e-12
INF = 1e10


@DATASETS.register_module()
class KittiMonoDatasetMonoCon(KittiMonoDataset):

    def __init__(self,
                 data_root,
                 info_file,
                 min_height=EPS,
                 min_depth=EPS,
                 max_depth=INF,
                 max_truncation=INF,
                 max_occlusion=INF,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            info_file=info_file,
            **kwargs)

        self.min_height = min_height
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_truncation = max_truncation
        self.max_occlusion = max_occlusion

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        add filtering mechanism based on occlusion, truncation or depth compared with its superclass

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_bboxes_cam3d_cam0 = []
        gt_bboxes_cam3d_cam2 = []
        centers2d = []
        depths = []

        assert len(ann_info) > 0
        has_kpt = ann_info[0].get('keypoints', None)

        if has_kpt is not None:
            has_kpt = True
            gt_kpts_2d = []
            gt_kpts_valid_mask = []
        else:
            has_kpt = False
            gt_kpts_2d = None
            gt_kpts_valid_mask = None

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            x2 = min(img_info['width'], x1 + w)
            y2 = min(img_info['height'], y1 + h)
            x1 = max(0, x1)
            y1 = max(0, y1)
            bbox = [x1, y1, x2, y2]
            # 3D annotations in camera coordinates (cam2 or cam0)
            bbox_cam3d_cam2 = np.array(ann['bbox_cam3d']).reshape(-1, )
            bbox_cam3d_cam0 = np.array(ann['bbox_cam3d_cam0']).reshape(-1, )

            if ann.get('iscrowd', False) or ann['occluded'] > self.max_occlusion \
                    or ann['truncated'] > self.max_truncation or ann['center2d'][2] > self.max_depth or \
                    ann['center2d'][2] < self.min_depth or (y2 - y1) < self.min_height:
                gt_bboxes_ignore.append(bbox)
                continue

            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['category_id']])
            gt_masks_ann.append(ann.get('segmentation', None))
            # 3D annotations in camera coordinates
            gt_bboxes_cam3d_cam0.append(bbox_cam3d_cam0)
            gt_bboxes_cam3d_cam2.append(bbox_cam3d_cam2)
            # 2.5D annotations in camera coordinates
            center2d = ann['center2d'][:2]
            depth = ann['center2d'][2]
            centers2d.append(center2d)
            depths.append(depth)

            # projected keypoints
            if has_kpt:
                kpts_2d = np.array(ann['keypoints']).reshape(-1, 3)
                kpts_valid_mask = kpts_2d[:, 2].astype('int64')
                kpts_2d = kpts_2d[:, :2].astype('float32').reshape(-1)

                gt_kpts_2d.append(kpts_2d)
                gt_kpts_valid_mask.append(kpts_valid_mask)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            if has_kpt:
                gt_kpts_2d = np.array(gt_kpts_2d)
                gt_kpts_valid_mask = np.array(gt_kpts_valid_mask)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            if has_kpt:
                gt_kpts_2d = np.array([], dtype=np.float32)
                gt_kpts_valid_mask = np.array([], dtype=np.int64)

        if gt_bboxes_cam3d_cam0:
            gt_bboxes_cam3d_cam0 = np.array(
                gt_bboxes_cam3d_cam0, dtype=np.float32)  # (0.5, 0.5, 0.5)
            gt_bboxes_cam3d_cam2 = np.array(
                gt_bboxes_cam3d_cam2, dtype=np.float32)  # (0.5, 0.5, 0.5)
            centers2d = np.array(centers2d, dtype=np.float32)
            depths = np.array(depths, dtype=np.float32)
        else:
            gt_bboxes_cam3d_cam0 = np.zeros(
                (0, self.bbox_code_size), dtype=np.float32)
            gt_bboxes_cam3d_cam2 = np.zeros(
                (0, self.bbox_code_size), dtype=np.float32)
            centers2d = np.zeros((0, 2), dtype=np.float32)
            depths = np.zeros((0), dtype=np.float32)

        box3d_dim = gt_bboxes_cam3d_cam0.shape[-1]
        gt_bboxes_cam3d_cam0 = CameraInstance3DBoxes(
            gt_bboxes_cam3d_cam0,
            box_dim=box3d_dim,
            origin=(0.5, 0.5, 0.5))  # change to (0.5, 1.0, 0.5)
        gt_bboxes_cam3d_cam2 = CameraInstance3DBoxes(
            gt_bboxes_cam3d_cam2,
            box_dim=box3d_dim,
            origin=(0.5, 0.5, 0.5))  # change to (0.5, 1.0, 0.5)
        gt_labels_3d = copy.deepcopy(gt_labels)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)


        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_bboxes_3d=gt_bboxes_cam3d_cam0,
            gt_bboxes_3d_cam2=gt_bboxes_cam3d_cam2,
            gt_labels_3d=gt_labels_3d,
            centers2d=centers2d,
            depths=depths,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        if has_kpt:
            ann.update(
                dict(gt_kpts_2d=gt_kpts_2d,
                     gt_kpts_valid_mask=gt_kpts_valid_mask,
                     )
            )

        return ann
