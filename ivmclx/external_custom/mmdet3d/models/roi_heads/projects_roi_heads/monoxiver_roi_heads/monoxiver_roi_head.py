import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from mmdet.core import bbox2roi, bbox_xyxy_to_cxcywh, build_assigner, build_sampler
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead


@HEADS.register_module()
class MonoXiverRoIHead(StandardRoIHead):
    def __init__(self,
                 bbox_roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                     out_channels=64,
                     featmap_strides=[4]),
                 num_augments=25,
                 bbox_head=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert mask_head is None
        super(MonoXiverRoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.num_augments = num_augments

    def forward_train(self,
                      x,
                      proposals_dict,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_3d,
                      **kwargs):
        proposals_bboxes_2d = proposals_dict['proposals_bboxes_2d']
        proposals_bboxes_3d = proposals_dict['proposals_bboxes_3d']
        proposals_labels = proposals_dict['proposals_labels']
        proposals_scores = proposals_dict['proposals_scores']
        proposals_projected_corners = proposals_dict['proposals_projected_corners']
        proposals_scores_logits = proposals_dict['proposals_scores_logits']

        num_imgs = len(img_metas)
        sampling_results = []
        for i in range(num_imgs):
            # use 3d to assign pos bboxes
            bboxes_3d_pred = torch.cat(proposals_bboxes_3d[i], dim=0)
            bboxes_2d_pred = torch.cat(proposals_bboxes_2d[i], dim=0)
            scores_logits = torch.cat(proposals_scores_logits[i], dim=0)

            num_proposal = len(bboxes_3d_pred)

            meta = img_metas[i]
            h, w = meta['pad_shape'][:2]
            img_whwh = x[0].new_tensor([[w, h] * 2]).repeat(num_proposal, 1)

            device = bboxes_3d_pred.device
            gt_bbox_3d = gt_bboxes_3d[i]
            if not isinstance(gt_bbox_3d, torch.Tensor):
                gt_bbox_3d = gt_bbox_3d.tensor.to(device)
                gt_bbox_3d[:, 1] -= 0.5 * gt_bbox_3d[:, 4]  # to (0.5, 0.5, 0.5)
                gt_bboxes_3d[i] = gt_bbox_3d

            normalize_bbox_ccwh = bbox_xyxy_to_cxcywh(bboxes_2d_pred / img_whwh)

            assign_result = self.bbox_assigner.assign(
                normalize_bbox_ccwh,
                bboxes_3d_pred,
                scores_logits,
                gt_bboxes[i],
                gt_bbox_3d,
                gt_labels[i],
                meta)

            sampling_result = self.bbox_sampler.sample(
                assign_result,
                bboxes_2d_pred,
                gt_bboxes[i])

            sampling_results.append(sampling_result)

        losses = dict()
        bbox_result = self._bbox_forward_train(x,
                                               proposals_bboxes_3d,
                                               proposals_bboxes_2d,
                                               proposals_projected_corners,
                                               sampling_results,
                                               gt_bboxes,
                                               gt_labels,
                                               gt_bboxes_3d,
                                               img_metas)
        losses.update(bbox_result['loss_bbox_3d'])

        return losses

    def _bbox_forward_train(self, 
                            x,
                            proposals_bboxes_3d,
                            proposals_bboxes_2d,
                            proposals_projected_corners,
                            sampling_results,
                            gt_bboxes,
                            gt_labels,
                            gt_bboxes_3d,
                            img_metas):
        bbox_result = self._bbox_forward(x,
                                        proposals_bboxes_3d, 
                                        proposals_bboxes_2d, 
                                        proposals_projected_corners,
                                        img_metas)

        bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes,
            gt_labels,
            gt_bboxes_3d,
            proposals_bboxes_3d,)
        loss_bbox = self.bbox_head.loss(bbox_result['cls_score'],
                                    bbox_result['xyz'],
                                    bbox_result['dim'],
                                    *bbox_targets)

        bbox_result.update(dict(loss_bbox_3d=loss_bbox))
        return bbox_result

    def _sample_corner_features(self, feat_map_levels, proposals_projected_corners, img_metas):
        feat_map_level = feat_map_levels[0]  # use highest resolution level to extract point features
        corners_feats = []
        for feat_map, corners, meta in zip(feat_map_level, proposals_projected_corners, img_metas):
            h, w = meta['pad_shape'][:2]
            corners = torch.cat(corners, dim=0)
            feat_map = feat_map.unsqueeze(0)
            mapping_size = feat_map.new_tensor([w, h]).view(1, 1, 1, 2)
            n_proposal = corners.shape[0]

            corners = corners.reshape(1, n_proposal, 9, 2)

            normalized_corners = corners / mapping_size
            normalized_corners = normalized_corners * 2.0 - 1.0
            normalized_corners = normalized_corners.clamp(min=-1, max=1)
            out = F.grid_sample(feat_map, normalized_corners,
                                mode='bilinear', padding_mode='zeros', align_corners=True)  # out: 1, C, n_proposal, n_kpts
            out = out[0].permute(1, 2, 0)  # out: n_proposal, n_kpts, C
            corners_feats.append(out)
        return corners_feats

    def _bbox_forward(self, x,
                    proposals_bboxes_3d,
                    proposals_bboxes_2d,
                    proposals_projected_corners,
                    img_metas,
                    debug=False):
        rois = bbox2roi([torch.cat(res, dim=0) for res in proposals_bboxes_2d])
        b = x[0].shape[0]

        # n, c, r_size, r_size
        roi_feats_all = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        roi_feats = []
        for i in range(b):
            # print(i)
            roi_feats.append(roi_feats_all[rois[:, 0] == i])

        bbox_geo_feats = []
        for b2d, b3d, bc, meta in zip(proposals_bboxes_2d, proposals_bboxes_3d, proposals_projected_corners, img_metas):
            h, w = meta['pad_shape'][:2]

            b2d = torch.cat(b2d, dim=0)
            b3d = torch.cat(b3d, dim=0)[..., :7]
            bc = torch.cat(bc, dim=0)

            b2d[..., 0::2] /= w
            b2d[..., 1::2] /= h
            b2d = b2d.clamp(max=1, min=0)
            bc[..., 0::2] /= w
            bc[..., 1::2] /= h
            bc = bc.clamp(max=1, min=0)

            b3d[..., 0] /= 50.
            b3d[..., 1] /= 2.
            b3d[..., 2] /= 80.

            bbox_geo_feats.append(torch.cat([b2d, b3d, bc], dim=-1))

        corners_feats = self._sample_corner_features(x, proposals_projected_corners, img_metas)
        # predict residual
        cls_score, xyz, dim = self.bbox_head(roi_feats, bbox_geo_feats, corners_feats)

        bbox_result = dict(cls_score=cls_score, xyz=xyz, dim=dim)

        return bbox_result

    def gather_topk(self, preds, indices):
        assert len(preds.shape) in (2, 3)
        assert len(indices.shape) == 2
        if len(preds.shape) == 2:
            return preds.gather(1, indices)
        else:
            _, _, c = preds.shape
            indices = indices.unsqueeze(2).repeat(1, 1, c)
            return preds.gather(1, indices)

    def filter_augment_box(self,
                           proposal_bboxes_3d,
                           proposal_labels,
                           proposal_scores,
                           cls,
                           scores,
                           xyz_res,
                           dim_res):
        num_proposals = len(proposal_bboxes_3d) // self.num_augments
        box_3d_size = proposal_bboxes_3d.shape[-1]

        proposal_bboxes_3d = proposal_bboxes_3d.reshape(num_proposals, self.num_augments, box_3d_size)
        proposal_labels = proposal_labels.reshape(num_proposals, self.num_augments)
        initial_scores = proposal_scores.reshape(num_proposals, self.num_augments)
        scores = scores.reshape(num_proposals, self.num_augments)
        xyz_res = xyz_res.reshape(num_proposals, self.num_augments, 3)
        dim_res = dim_res.reshape(num_proposals, self.num_augments, 3)
        cls = cls.reshape(num_proposals, self.num_augments)

        topk_num = self.test_cfg.get('topk', 8)
        topk_scores, topk_indices = scores.topk(topk_num, dim=1)

        box3d_topk = self.gather_topk(proposal_bboxes_3d, topk_indices.clone()).reshape(num_proposals * topk_num, box_3d_size)
        initial_scores_topk = self.gather_topk(initial_scores, topk_indices.clone()).reshape(num_proposals * topk_num)
        label_topk = self.gather_topk(proposal_labels, topk_indices.clone()).reshape(num_proposals * topk_num)
        cls_topk = self.gather_topk(cls, topk_indices.clone()).reshape(num_proposals * topk_num)

        topk_scores = topk_scores.reshape(num_proposals * topk_num)

        xyz_res_topk = self.gather_topk(xyz_res, topk_indices.clone()).reshape(num_proposals * topk_num, 3)
        dim_res_topk = self.gather_topk(dim_res, topk_indices.clone()).reshape(num_proposals * topk_num, 3)

        mask = topk_scores > self.test_cfg.score_thr_post

        box3d_pred = box3d_topk[mask]
        label_pred = label_topk[mask]
        cls_pred = cls_topk[mask]
        xyz_res_pred = xyz_res_topk[mask]
        dim_res_pred = dim_res_topk[mask]

        final_rescore_topk = initial_scores_topk * topk_scores
        final_rescore_topk = final_rescore_topk[mask]
        score_pred = final_rescore_topk

        mask_final = score_pred > self.test_cfg.score_thr_final
        box3d_pred = box3d_pred[mask_final]
        label_pred = label_pred[mask_final]
        cls_pred = cls_pred[mask_final]
        xyz_res_pred = xyz_res_pred[mask_final]
        dim_res_pred = dim_res_pred[mask_final]
        score_pred = score_pred[mask_final]

        return box3d_pred, label_pred, score_pred, xyz_res_pred, dim_res_pred

    def simple_test(self,
                    x,
                    proposals_dict,
                    img_metas,
                    rescale=False):
        box_type_3d = img_metas[0]['box_type_3d']

        # parse proposals dict
        proposal_scores = proposals_dict['proposals_scores']
        assert len(proposal_scores) == 1
        proposal_scores = proposal_scores[0]
        proposal_bboxes_2d = proposals_dict['proposals_bboxes_2d'][0]
        proposal_bboxes_3d = proposals_dict['proposals_bboxes_3d'][0]
        proposal_labels = proposals_dict['proposals_labels'][0]
        proposal_projected_corners = proposals_dict['proposals_projected_corners'][0]

        num_raw_proposal = len(proposal_scores[0])
        num_augment_proposal = len(proposal_scores[1])
        num_proposals = num_raw_proposal + num_augment_proposal

        if num_proposals == 0:
            det_results = [
                [box_type_3d(proposal_bboxes_3d[0],
                             box_dim=proposal_bboxes_3d[0].shape[-1], origin=(0.5, 0.5, 0.5)),
                 proposal_scores[0],
                 proposal_labels[0],
                 ]
            ]
            return det_results

        bbox_result = self._bbox_forward(x, 
                                        [proposal_bboxes_3d],
                                        [proposal_bboxes_2d],
                                        [proposal_projected_corners],
                                        img_metas,
                                        )

        scores = bbox_result['cls_score']
        cls = scores.argmax(dim=1)
        scores = scores.sigmoid().max(dim=1)[0]
        xyz_res = bbox_result['xyz']
        dim_res = bbox_result['dim']

        if num_augment_proposal > 0:
            # post-filter augment boxes
            aug_box3d_pred, aug_label_pred, \
            aug_scores_pred, aug_xyz_res_pred, aug_dim_res_pred \
             = self.filter_augment_box(
                    proposal_bboxes_3d[1],
                    proposal_labels[1],
                    proposal_scores[1],
                    cls[num_raw_proposal:],
                    scores[num_raw_proposal:],
                    xyz_res[num_raw_proposal:],
                    dim_res[num_raw_proposal:])

            box3d_pred = torch.cat([proposal_bboxes_3d[0], aug_box3d_pred], dim=0)
            label_pred = torch.cat([proposal_labels[0], aug_label_pred], dim=0)
            score_pred = torch.cat([scores[:num_raw_proposal] * proposal_scores[0], aug_scores_pred], dim=0)
            xyz_res_pred = torch.cat([xyz_res[:num_raw_proposal], aug_xyz_res_pred], dim=0)
            dim_res_pred = torch.cat([dim_res[:num_raw_proposal], aug_dim_res_pred], dim=0)
        else:
            box3d_pred = proposal_bboxes_3d[0]
            score_pred = scores * proposal_scores[0]
            # score_pred = proposal_scores[0]
            label_pred = proposal_labels[0]
            xyz_res_pred = xyz_res
            dim_res_pred = dim_res

        box3d_pred[..., :3] += xyz_res_pred
        box3d_pred[..., 3:6] += dim_res_pred

        det_results = [
            [box_type_3d(box3d_pred,
                         box_dim=box3d_pred.shape[-1], origin=(0.5, 0.5, 0.5)),
             score_pred,
             label_pred,
             ]
        ]

        return det_results

    def forward_dummy(self, x, proposals_dict,
                    img_metas,):
        proposals_bboxes_2d = proposals_dict['proposals_bboxes_2d']
        proposals_bboxes_3d = proposals_dict['proposals_bboxes_3d']
        proposals_projected_corners = proposals_dict['proposals_projected_corners']

        bbox_result = self._bbox_forward(x,
                                        proposals_bboxes_3d, 
                                        proposals_bboxes_2d, 
                                        proposals_projected_corners,
                                        img_metas)

        return bbox_result
