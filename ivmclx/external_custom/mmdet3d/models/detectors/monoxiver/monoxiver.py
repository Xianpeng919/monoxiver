import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

import torch.nn.functional as F

from mmdet.models.builder import DETECTORS, build_head
from mmdet.models.detectors.two_stage import TwoStageDetector

from mmdet3d.core import (CameraInstance3DBoxes, bbox3d2result)


@DETECTORS.register_module()
class MonoXiver(TwoStageDetector):
    def __init__(self, *args,
                 refine_only=True,
                 proposal_generator=None,          
                 **kwargs):
        super(MonoXiver, self).__init__(*args, **kwargs)
        self.refine_only = refine_only
        assert refine_only
        self.proposal_generator = build_head(proposal_generator)

    def train(self, mode=True):
        super(MonoXiver, self).train(mode)
        # set rpn head to eval mode by default
        if self.refine_only:
            for modules in [self.backbone, self.neck, self.rpn_head]:
                for m in modules.modules():
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      centers2d=None,
                      depths=None,
                      gt_kpts_2d=None,
                      gt_kpts_valid_mask=None,
                      **kwargs):
        losses = {}
        feats = self.extract_feat(img)

        # for first stage detection
        rpn_result = self.rpn_head.forward_train(feats,
                    img_metas)
        proposals_list = rpn_result['proposals_list']

        proposals_dict = self.proposal_generator(proposals_list, img_metas)
        # 2nd stage refinement
        roi_losses = self.roi_head.forward_train(
            feats,
            proposals_dict,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            **kwargs
        )

        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        num_imgs = len(img_metas)
        assert num_imgs == 1

        feats = self.extract_feat(img)
        
        # for first stage detection
        proposals_list = self.rpn_head.forward_test(feats, img_metas)
        proposals_dict = self.proposal_generator(proposals_list, img_metas)

        # 2nd stage refinement
        results = self.roi_head.simple_test(
            feats,
            proposals_dict,
            img_metas,
            rescale=rescale)

        bbox_img = []
        for bbox_output in results:
            bboxes, scores, labels = bbox_output[:3]
            bbox_img.append(bbox3d2result(bboxes, scores, labels))

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox

        return bbox_list

    def forward_dummy(self, img):
        outs = ()
        b, _, h, w = img.shape

        x = self.extract_feat(img)
        rpn_outs = self.rpn_head(x)
        outs = outs + (rpn_outs, )

        proposals_list = []
        img_metas = []
        for _ in range(b):
            proposal = (
                torch.rand(15, 7).cuda(),
                torch.zeros(15, 1).long().cuda(),
                torch.rand(15, 1).cuda(),
            )
            proposals_list.append(proposal)

            img_meta = {}
            img_meta['cam2img'] = np.array([[w/2, 10, 0, 0], [10, h/2, 0, 0], [0, 0, 1, 0]])
            img_meta['box_type_3d'] = CameraInstance3DBoxes
            img_meta['pad_shape'] = (384, 1248)
            img_metas.append(img_meta)

        proposals_dict = self.proposal_generator(proposals_list, img_metas)

        # batch_bboxes_3d, batch_topk_labels, batch_scores, score_logits
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals_dict, img_metas)
        outs = outs + (roi_outs, )
        return outs