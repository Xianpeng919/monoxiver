import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmdet.core import multi_apply, build_bbox_coder
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_transformer
from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead

from mmdet.core.bbox.iou_calculators.builder import build_iou_calculator


import numpy as np
PI = np.pi


@HEADS.register_module()
class MonoXiverBboxHead(BBoxHead):

    def __init__(self,
                 with_cls=True,
                 with_reg=True,
                 num_classes=3,
                 in_channels=64,
                 feat_channel=256,
                 num_sa_layers=1,
                 num_modes=1,
                 num_heads=8,
                 num_ffn_fcs=2,
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 feedforward_channels=2048,
                 dropout=0.0,
                 num_cls_fcs=2,
                 num_reg_fcs=2,
                 loss_xyz=dict(type='L1Loss', loss_weight=5.0),
                 loss_dim=dict(type='L1Loss', loss_weight=1.0),
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(MonoXiverBboxHead, self).__init__(
            num_classes=num_classes,
            with_cls=with_cls,
            with_reg=with_reg,
            in_channels=in_channels,
            **kwargs)
        self.fp16_enabled = False

        self.loss_xyz = build_loss(loss_xyz)
        self.loss_dim = build_loss(loss_dim)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channel = feat_channel

        self.num_modes = num_modes
        self.ffn_act_cfg = ffn_act_cfg

        # init geometric embedding
        box_2d_embed_channels = 4
        box_3d_embed_channels = 7
        projection_feat_channels = 18
        geometric_channels = box_2d_embed_channels + box_3d_embed_channels + projection_feat_channels

        self.geometric_embedding = self._build_reg_fc(1, geometric_channels, feat_channel * num_modes)
        self.corner_feat_embedding = self._build_reg_fc(1, in_channels, feat_channel)

        # geometric & corner feat interaction
        self.geo_mhca = MultiheadAttention(feat_channel, num_heads)
        self.geo_ca_norm = build_norm_layer(dict(type='LN'), feat_channel)[1]
        self.geo_ca_ffn = FFN(
            feat_channel,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.geo_ca_ffn_norm = build_norm_layer(dict(type='LN'), feat_channel)[1]

        self.num_sa_layers = num_sa_layers
        assert self.num_sa_layers > 0
        for i in range(num_sa_layers):
            setattr(self, f'geo_mhsa_{i}', MultiheadAttention(feat_channel, num_heads, dropout))
            setattr(self, f'geo_sa_norm_{i}', build_norm_layer(dict(type='LN'), feat_channel)[1])
            setattr(self, f'geo_sa_ffn_{i}', FFN(
                                                feat_channel,
                                                feedforward_channels,
                                                num_ffn_fcs,
                                                act_cfg=ffn_act_cfg,
                                                dropout=dropout
                                            )
                    )
            setattr(self, f'geo_sa_ffn_norm_{i}', build_norm_layer(dict(type='LN'), feat_channel)[1])

        img_in_channels = self.in_channels * self.roi_feat_area
        self.img_embed = self._build_reg_fc(1, img_in_channels, feat_channel)

        # decode image embedding
        self.decode_mhca = MultiheadAttention(feat_channel, num_heads)
        self.decode_ca_norm = build_norm_layer(dict(type='LN'), feat_channel)[1]
        self.decode_ca_ffn = FFN(
            feat_channel,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.decode_ca_ffn_norm = build_norm_layer(dict(type='LN'), feat_channel)[1]        

        self.attention = MultiheadAttention(feat_channel, num_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), feat_channel)[1]

        self.ffn = FFN(
            feat_channel,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), feat_channel)[1]

        self.cls_fc = self._build_reg_fc(num_cls_fcs, feat_channel, feat_channel)
        self.dim_fc = self._build_reg_fc(num_reg_fcs, feat_channel, feat_channel)
        self.xyz_fc = self._build_reg_fc(num_reg_fcs, feat_channel, feat_channel)

        # over load the self.fc_cls in BBoxHead,
        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(feat_channel, self.num_classes)
        else:
            self.fc_cls = nn.Linear(feat_channel, self.num_classes + 1)

        self.dim = nn.Linear(feat_channel, 3)
        self.xyz = nn.Linear(feat_channel, 3)

    def _build_reg_fc(self, num_reg_fcs, in_channels, feat_channels):
        layer_list = []
        for _ in range(num_reg_fcs):
            layer_list.append(
                nn.Linear(in_channels, feat_channels, bias=False))
            layer_list.append(build_norm_layer(dict(type='LN'), feat_channels)[1])
            layer_list.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))
            in_channels = feat_channels
        layer_list = nn.Sequential(*layer_list)
        return layer_list

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        super(MonoXiverBboxHead, self).init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)
    
    @auto_fp16()
    def encode_corner_feats(self, corner_feat, bbox_geo_feat):
        corner_feat_embed = self.corner_feat_embedding(corner_feat).permute(1, 0, 2)  # n_kpts, b, c
        _, n_proposal, c = corner_feat_embed.shape
        
        geo_params_embed = self.geometric_embedding(bbox_geo_feat).reshape(n_proposal, self.num_modes, self.feat_channel)  # b, c*n_modes -> b, n_modes, c
        geo_params_embed = geo_params_embed.permute(1, 0, 2)  # n_modes, b, c

        geo_feat = self.geo_ca_norm(self.geo_mhca(query=geo_params_embed, 
                                                    key=corner_feat_embed, 
                                                    value=corner_feat_embed))  # n_mode, b, c

        geo_feat = geo_feat.permute(1, 0, 2)  # b, n_kpts, c
        geo_feat = self.geo_ca_ffn_norm(self.geo_ca_ffn(geo_feat))

        for i in range(self.num_sa_layers):
            geo_mhsa = getattr(self, f'geo_mhsa_{i}')      
            geo_sa_norm = getattr(self, f'geo_sa_norm_{i}')
            geo_sa_ffn = getattr(self, f'geo_sa_ffn_{i}') 
            geo_sa_ffn_norm = getattr(self, f'geo_sa_ffn_norm_{i}') 

            geo_feat = geo_feat.permute(1, 0, 2)  # n_kpts, b, c
            geo_feat = geo_sa_norm(geo_mhsa(geo_feat))
            geo_feat = geo_feat.permute(1, 0, 2)  # b, n_kpts, c
            geo_feat = geo_sa_ffn_norm(geo_sa_ffn(geo_feat))

        return geo_feat

    @auto_fp16()
    def decode_image_embedding(self, corner_feat, img_feat):
        img_feat = img_feat.unsqueeze(0)  # 1, b, c
        corner_feat = corner_feat.permute(1, 0, 2)  # n_kpt, b, c

        decode_feat = self.decode_ca_norm(self.decode_mhca(query=img_feat,   # 1, b, c
                                                        key=corner_feat,   # n_kpt, b, c
                                                        value=corner_feat,   # n_kpt, b, c
                                                    )
                                                )  # 1, b, c
        decode_feat = decode_feat.permute(1, 0, 2)  # b, 1, c
        decode_feat = self.decode_ca_ffn_norm(self.geo_ca_ffn(decode_feat))  # b, 1, c
        return decode_feat

    @auto_fp16()
    def forward(self, roi_feats, bbox_geo_feats, corners_feats):
        # self attention is calculated with embeddings within one image
        object_feat = []

        for b in range(len(roi_feats)):
            roi_feat = roi_feats[b]
            bbox_geo_feat = bbox_geo_feats[b]
            corners_feat = corners_feats[b]  # n_proposal, 9, 64

            # appearance feat
            img_feat = self.img_embed(roi_feat.flatten(1))  # n_proposal, self.feat_channel

            # corner feat
            corners_feat = self.encode_corner_feats(corners_feat, bbox_geo_feat)

            # decode corner feat
            object_feat_single = self.decode_image_embedding(corners_feat, img_feat)  # b, 1, c

            object_feat_single = self.attention_norm(self.attention(object_feat_single))
            object_feat_single = object_feat_single.permute(1, 0, 2).squeeze(0)
            object_feat.append(object_feat_single)

        obj_feat = torch.cat(object_feat)

        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))  # 800, 320

        cls_feat = obj_feat
        dim = obj_feat
        xyz = obj_feat

        cls_feat = self.cls_fc(cls_feat)
        dim = self.dim_fc(dim)
        xyz = self.xyz_fc(xyz)

        xyz = self.xyz(xyz)
        dim = self.dim(dim)

        cls_score = self.fc_cls(cls_feat).view(
            -1, self.num_classes
            if self.loss_cls.use_sigmoid else self.num_classes + 1)

        return cls_score, xyz, dim

    @force_fp32(apply_to=('cls_score', 'xyz_pred', 'dim_pred'))
    def loss(self,
             cls_score,
             xyz_pred,
             dim_pred,
             labels,
             label_weights,
             xyz_target,
             dim_target,
             alpha_target,
             reduction_override=None,
             **kwargs):
        losses = dict()
        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = max(pos_inds.sum().float(), 1)
        avg_factor = reduce_mean(num_pos)
        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['pos_acc'] = accuracy(cls_score[pos_inds],
                                             labels[pos_inds])

        if xyz_pred is not None:
            if pos_inds.any():
                losses['loss_xyz_refine'] = self.loss_xyz(xyz_pred[pos_inds], xyz_target[pos_inds])
                losses['loss_dim_refine'] = self.loss_dim(dim_pred[pos_inds], dim_target[pos_inds])
            else:
                losses['loss_xyz_refine'] = xyz_pred.sum() * 0
                losses['loss_dim_refine'] = xyz_pred.sum() * 0

        return losses

    def _get_target_single(self, pos_inds, neg_inds,
                           pos_bboxes, neg_bboxes,
                           pos_gt_bboxes,
                           pos_gt_labels,
                           pos_gt_xyz,
                           pos_gt_dim_3d,
                           pos_gt_orientation,
                           cfg=None):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.
        Almost the same as the implementation in `bbox_head`,
        we add pos_inds and neg_inds to select positive and
        negative samples instead of selecting the first num_pos
        as positive samples.
        Args:
            pos_inds (Tensor): The length is equal to the
                positive sample numbers contain all index
                of the positive sample in the origin proposal set.
            neg_inds (Tensor): The length is equal to the
                negative sample numbers contain all index
                of the negative sample in the origin proposal set.
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains all the gt_boxes,
                has shape (num_gt, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains all the gt_labels,
                has shape (num_gt).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.
        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following Tensors:
                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all proposals, has
                  shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all proposals, has
                  shape (num_proposals, 4), the last dimension 4
                  represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all proposals,
                  has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples,),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        dim_3d_target = pos_bboxes.new_zeros(num_samples, 3)
        xyz_target = pos_bboxes.new_zeros(num_samples, 3)
        orientation_target = pos_bboxes.new_zeros(num_samples, 1)
        if num_pos > 0:
            if self.num_classes == 1:
                labels[pos_inds] = 0.
            else:
                labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0
            label_weights[pos_inds] = pos_weight

            dim_3d_target[pos_inds, :] = pos_gt_dim_3d
            xyz_target[pos_inds, :] = pos_gt_xyz
            orientation_target[pos_inds, :] = pos_gt_orientation

        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        return labels, label_weights, xyz_target, dim_3d_target, orientation_target

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    gt_bboxes_3d,
                    proposals_bbox3d,
                    rcnn_train_cfg=None,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.
        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.
        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:`ConfigDict`): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.
        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:
                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise just
                  a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals,) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list has
                  shape (num_proposals, 4) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals, 4),
                  the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]

        pos_gt_dim_3d_list = []
        pos_gt_xyz_list = []
        pos_gt_orientation_list = []

        for i in range(len(sampling_results)):
            res = sampling_results[i]
            pos_assigned_gt_inds = res.pos_assigned_gt_inds
            pos_inds = pos_inds_list[i]

            gt_bbox_3d = gt_bboxes_3d[i]
            assert isinstance(gt_bbox_3d, torch.Tensor)
            proposal_bbox3d = torch.cat(proposals_bbox3d[i], dim=0)

            dim = gt_bbox_3d[pos_assigned_gt_inds, 3: 6] - proposal_bbox3d[pos_inds, 3: 6]
            xyz = gt_bbox_3d[pos_assigned_gt_inds, 0: 3] - proposal_bbox3d[pos_inds, 0: 3]
            alpha = (gt_bbox_3d[pos_assigned_gt_inds, 6:7] - proposal_bbox3d[pos_inds, 6: 7]) % (2 * PI)

            pos_gt_dim_3d_list.append(dim)
            pos_gt_xyz_list.append(xyz)
            pos_gt_orientation_list.append(alpha)

        labels, label_weights, xyz_targets, dim_3d_targets, orientation_targets = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            pos_gt_xyz_list,
            pos_gt_dim_3d_list,
            pos_gt_orientation_list,
            cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)  # 800
            label_weights = torch.cat(label_weights, 0)  # 800
            xyz_targets = torch.cat(xyz_targets, 0)  # 800, 1
            dim_3d_targets = torch.cat(dim_3d_targets, 0)  # 800, 3
            orientation_targets = torch.cat(orientation_targets, 0)

        return labels, label_weights, xyz_targets, dim_3d_targets, orientation_targets
