import os.path as osp
import numpy as np

from skimage import io

from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines.loading import LoadAnnotations3D


@PIPELINES.register_module()
class LoadAnnotations3DMonoCon(LoadAnnotations3D):
    def __init__(self,
                 with_2D_kpts=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.with_2D_kpts = with_2D_kpts

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results = super()._load_bboxes_3d(results)
        results['gt_bboxes_3d_cam2'] = results['ann_info']['gt_bboxes_3d_cam2']
        results['bbox3d_fields'].append('gt_bboxes_3d_cam2')
        return results

    def _load_kpts_2d(self, results):
        results['gt_kpts_2d'] = results['ann_info']['gt_kpts_2d']
        results['gt_kpts_valid_mask'] = results['ann_info']['gt_kpts_valid_mask']
        return results

    def __call__(self, results):
        results = super().__call__(results)
        if self.with_2D_kpts:
            results = self._load_kpts_2d(results)

        return results
