from os import path as osp
import numpy as np
import torch
import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.models.builder import DETECTORS
from mmdet3d.models.detectors.single_stage_mono3d import SingleStageMono3DDetector
from mmdet3d.core import (CameraInstance3DBoxes, bbox3d2result, show_multi_modality_result)


@DETECTORS.register_module()
class CenterNetMono3D(SingleStageMono3DDetector):

    def __init__(self,
                 *args,
                 **kwargs):
        self.use_2d_box_in_3d = kwargs.get('use_2d_box_in_3d', False)
        super(CenterNetMono3D, self).__init__(*args, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      centers2d,
                      depths,
                      **kwargs):
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels,
                                              gt_bboxes_3d=gt_bboxes_3d,
                                              gt_labels_3d=gt_labels_3d,
                                              centers2d=centers2d,
                                              depths=depths,
                                              **kwargs)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_outputs = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        if not self.bbox_head.pred_bbox2d:
            bbox_img = []
            for bbox_output in bbox_outputs:
                bboxes, scores, labels = bbox_output
                bbox_img.append(bbox3d2result(bboxes, scores, labels))
        elif self.use_2d_box_in_3d:
            from mmdet3d.core import bbox2d3d2result
            bbox_img = []
            for bbox_output in bbox_outputs:
                bboxes, scores, labels, bboxes2d = bbox_output
                bbox_img.append(bbox2d3d2result(bboxes, bboxes2d, scores, labels))
        else:
            from mmdet.core import bbox2result
            bbox2d_img = []
            bbox_img = []
            for bbox_output in bbox_outputs:
                bboxes, scores, labels, bboxes2d = bbox_output
                bbox2d_img.append(bbox2result(bboxes2d, labels, self.bbox_head.num_classes))
                bbox_img.append(bbox3d2result(bboxes, scores, labels))

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox
        if self.bbox_head.pred_bbox2d and not self.use_2d_box_in_3d:
            for result_dict, img_bbox2d in zip(bbox_list, bbox2d_img):
                result_dict['img_bbox2d'] = img_bbox2d
        return bbox_list

    def aug_test(self, imgs, img_metas, rescale=False):
        """Augment testing of CenterNet. Aug test must have flipped image pair,
        and unlike CornerNet, it will perform an averaging operation on the
        feature map instead of detecting bbox.

        Args:
            imgs (list[Tensor]): Augmented images.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.

        Note:
            ``imgs`` must including flipped image pairs.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        img_inds = list(range(len(imgs)))
        bboxes_3d_type = img_metas[0][0]['box_type_3d']
        assert img_metas[0][0]['flip'] + img_metas[1][0]['flip'], (
            'aug test must have flipped image pair')

        aug_results = []
        for ind, flip_ind in zip(img_inds[0::2], img_inds[1::2]):
            assert flip_ind == 1

            img_pair = torch.cat([imgs[ind], imgs[flip_ind]])
            x = self.extract_feat(img_pair)
            center_heatmap_preds, wh_preds, offset_preds, center2kpt_offset_preds, kpt_heatmap_preds, \
            kpt_heatmap_offset_preds, dim_preds, alpha_cls_preds, alpha_offset_preds, depth_preds = self.bbox_head(x)
            assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == \
                   len(center2kpt_offset_preds) == len(kpt_heatmap_preds) \
                   == len(kpt_heatmap_offset_preds) == len(dim_preds) == len(alpha_cls_preds) \
                   == len(alpha_offset_preds) == len(depth_preds) == 1

            center_heatmap_preds = center_heatmap_preds[0]
            wh_preds = wh_preds[0]
            offset_preds = offset_preds[0]
            center2kpt_offset_preds = center2kpt_offset_preds[0]
            kpt_heatmap_preds = kpt_heatmap_preds[0]
            kpt_heatmap_offset_preds = kpt_heatmap_offset_preds[0]
            dim_preds = dim_preds[0]
            alpha_cls_preds = alpha_cls_preds[0]
            alpha_offset_preds = alpha_offset_preds[0]
            depth_preds = depth_preds[0]

            bbox_list_wo_flip = self.bbox_head.get_bboxes(
                center_heatmap_preds[0:1],
                wh_preds[0:1],
                offset_preds[0:1],
                center2kpt_offset_preds[0:1],
                kpt_heatmap_preds[0:1],
                kpt_heatmap_offset_preds[0:1],
                dim_preds[0:1],
                alpha_cls_preds[0:1],
                alpha_offset_preds[0:1],
                depth_preds[0:1],
                img_metas[ind],
                rescale=rescale)

            bbox_list_with_flip = self.bbox_head.get_bboxes(
                center_heatmap_preds[1:2],
                wh_preds[1:2],
                offset_preds[1:2],
                center2kpt_offset_preds[1:2],
                kpt_heatmap_preds[1:2],
                kpt_heatmap_offset_preds[1:2],
                dim_preds[1:2],
                alpha_cls_preds[1:2],
                alpha_offset_preds[1:2],
                depth_preds[1:2],
                img_metas[ind],
                flipped=True,
                rescale=rescale)

            bbox_list = bbox_list_wo_flip + bbox_list_with_flip
            aug_results.append(bbox_list)

        with_nms = self.bbox_head.test_cfg.get('with_nms', False)
        assert with_nms
        # TODO: need debug
        bbox_outputs = [self.merge_aug_results(aug_results, bboxes_3d_type, self.bbox_head.test_cfg)]

        if not self.bbox_head.pred_bbox2d:
            bbox_img = []
            for bbox_output in bbox_outputs:
                bboxes, scores, labels = bbox_output
                bbox_img.append(bbox3d2result(bboxes, scores, labels))
        elif self.use_2d_box_in_3d:
            from mmdet3d.core import bbox2d3d2result
            bbox_img = []
            for bbox_output in bbox_outputs:
                bboxes, scores, labels, bboxes2d = bbox_output
                bbox_img.append(bbox2d3d2result(bboxes, bboxes2d, scores, labels))
        else:
            from mmdet.core import bbox2result
            bbox2d_img = []
            bbox_img = []
            for bbox_output in bbox_outputs:
                bboxes, scores, labels, bboxes2d = bbox_output
                bbox2d_img.append(bbox2result(bboxes2d, labels, self.bbox_head.num_classes))
                bbox_img.append(bbox3d2result(bboxes, scores, labels))

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox
        if self.bbox_head.pred_bbox2d and not self.use_2d_box_in_3d:
            for result_dict, img_bbox2d in zip(bbox_list, bbox2d_img):
                result_dict['img_bbox2d'] = img_bbox2d
        return bbox_list

    def merge_aug_results(self, aug_results, bboxes_3d_type, cfg):
        """Merge augmented detection bboxes and score.

        Args:
            aug_results (list[list[Tensor]]): Det_bboxes and det_labels of each
                image.
            with_nms (bool): If True, do nms before return boxes.

        Returns:
            tuple: (out_bboxes, out_labels)
        """
        assert len(aug_results) == 1
        thresh = cfg.thresh

        bboxes_3d = []
        bboxes_2d = []
        bboxes_labels = []
        bboxes_scores = []

        single_result = aug_results[0]
        results_wo_flip = single_result[0]
        results_w_flip = single_result[1]

        if not self.bbox_head.pred_bbox2d:
            bboxes_3d.append(results_wo_flip[0].tensor)
            bboxes_scores.append(results_wo_flip[1])
            bboxes_labels.append(results_wo_flip[2])
            bboxes_3d.append(results_w_flip[0].tensor)
            bboxes_scores.append(results_w_flip[1])
            bboxes_labels.append(results_w_flip[2])
        else:
            bboxes_3d.append(results_wo_flip[0].tensor)
            bboxes_scores.append(results_wo_flip[1])
            bboxes_labels.append(results_wo_flip[2])
            bboxes_2d.append(results_wo_flip[3])
            bboxes_3d.append(results_w_flip[0].tensor)
            bboxes_scores.append(results_w_flip[1])
            bboxes_labels.append(results_w_flip[2])
            bboxes_2d.append(results_w_flip[3])

        bboxes_3d = torch.cat(bboxes_3d, dim=0).contiguous()
        bboxes_labels = torch.cat(bboxes_labels).contiguous()
        bboxes_scores = torch.cat(bboxes_scores).contiguous()
        if self.bbox_head.pred_bbox2d:
            bboxes_2d = torch.cat(bboxes_2d, dim=0).contiguous()

        bboxes_3d_for_nms = xywhr2xyxyr(bboxes_3d_type(bboxes_3d, box_dim=bboxes_3d.shape[-1],
                                                          origin=(0.5, 0.5, 0.5)).bev)
        results = box3d_nms_monocon(bboxes_3d, bboxes_3d_for_nms,
                                    bboxes_labels, bboxes_scores,
                                    thresh, cfg.nms_post,
                                    cfg=cfg,
                                    bboxes2d=bboxes_2d if self.bbox_head.pred_bbox2d else None)

        if self.bbox_head.pred_bbox2d:
            bboxes_3d, bboxes_scores, bboxes_labels, bboxes_2d = results

            det_results = [
                [bboxes_3d_type(bboxes_3d,
                                box_dim=bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)),
                 bboxes_scores,
                 bboxes_labels,
                 bboxes_2d,
                 ]
            ]
        else:
            bboxes_3d, bboxes_scores, bboxes_labels = results
            det_results = [
                [bboxes_3d_type(bboxes_3d,
                                box_dim=bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)),
                 bboxes_scores,
                 bboxes_labels,
                 ]
            ]

        return det_results

    def show_results(self, data, result, out_dir, show=False, score_thr=None):
        """Results visualization.

        Args:
            data (list[dict]): Input images and the information of the sample.
            result (list[dict]): Prediction results.
            out_dir (str): Output directory of visualization result.
            show (bool, optional): Determines whether you are
                going to show result by open3d.
                Defaults to False.
            TODO: implement score_thr of single_stage_mono3d.
            score_thr (float, optional): Score threshold of bounding boxes.
                Default to None.
                Not implemented yet, but it is here for unification.
        """
        for batch_id in range(len(result)):
            if isinstance(data['img_metas'][0], DC):
                img_filename = data['img_metas'][0]._data[0][batch_id][
                    'filename']
                cam2img = data['img_metas'][0]._data[0][batch_id]['cam2img']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                img_filename = data['img_metas'][0][batch_id]['filename']
                cam2img = data['img_metas'][0][batch_id]['cam2img']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            img = mmcv.imread(img_filename)
            file_name = osp.split(img_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'

            pred_bboxes = result[batch_id]['img_bbox']['boxes_3d']
            assert isinstance(pred_bboxes, CameraInstance3DBoxes), \
                f'unsupported predicted bbox type {type(pred_bboxes)}'

            show_multi_modality_result(
                img,
                None,
                pred_bboxes,
                cam2img,
                out_dir,
                file_name,
                'camera',
                show=show)
