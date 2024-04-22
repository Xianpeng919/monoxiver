import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob, normal_init, kaiming_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils.gaussian_target import (gaussian_radius, gen_gaussian_target)
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                                transpose_and_gather_feat)
from ivmclx.models.backbones.layers.attentive_norm import AttnBatchNorm2d
from mmdet3d.core.bbox import points_cam2img

INF = 1e8
EPS = 1e-12
PI = np.pi


@HEADS.register_module()
class MonoConProposalHead(nn.Module):
    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 bbox3d_code_size=7,
                 num_kpt=9,
                 num_alpha_bins=12,
                 use_AN=False,
                 num_AN_affine=10,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(MonoConProposalHead, self).__init__()
        assert bbox3d_code_size >= 7
        self.num_classes = num_classes
        self.bbox_code_size = bbox3d_code_size
        self.pred_bbox2d = True
        self.max_objs = 30
        self.num_kpt = num_kpt
        self.num_alpha_bins = num_alpha_bins

        self.use_AN = use_AN
        self.num_AN_affine = num_AN_affine
        if use_AN:
            self.norm = AttnBatchNorm2d
        else: 
            self.norm = nn.BatchNorm2d

        self.heatmap_head = self._build_head(in_channel, feat_channel,
                                             num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)
        self.center2kpt_offset_head = self._build_head(in_channel, feat_channel, num_kpt * 2)
        self.kpt_heatmap_head = self._build_head(in_channel, feat_channel, num_kpt)
        self.kpt_heatmap_offset_head = self._build_head(in_channel, feat_channel, 2)
        self.dim_head = self._build_head(in_channel, feat_channel, 3)
        self.depth_head = self._build_head(in_channel, feat_channel, 2)
        self._build_dir_head(in_channel, feat_channel)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        norm_layer = self.norm(feat_channel, momentum=0.03, eps=0.001) if not self.use_AN else \
            self.norm(feat_channel, self.num_AN_affine,
                      momentum=0.03, eps=0.001)

        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            norm_layer,
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def _build_dir_head(self, in_channel, feat_channel):
        norm_layer = self.norm(feat_channel, momentum=0.03, eps=0.001) if not self.use_AN else \
            self.norm(feat_channel, self.num_AN_affine,
                      momentum=0.03, eps=0.001)

        self.dir_feat = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            norm_layer,
            nn.ReLU(inplace=True),
        )
        self.dir_cls = nn.Sequential(nn.Conv2d(feat_channel, self.num_alpha_bins, kernel_size=1))
        self.dir_reg = nn.Sequential(nn.Conv2d(feat_channel, self.num_alpha_bins, kernel_size=1))

    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)  # -2.19
        self.kpt_heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head, self.center2kpt_offset_head, self.depth_head,
                     self.kpt_heatmap_offset_head, self.dim_head, self.dir_feat, self.dir_cls,
                     self.dir_reg]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        center_heatmap_pred = torch.clamp(center_heatmap_pred, min=1e-4, max=1 - 1e-4)
        kpt_heatmap_pred = self.kpt_heatmap_head(feat).sigmoid()
        kpt_heatmap_pred = torch.clamp(kpt_heatmap_pred, min=1e-4, max=1 - 1e-4)

        offset_pred = self.offset_head(feat)
        kpt_heatmap_offset_pred = self.kpt_heatmap_offset_head(feat)

        wh_pred = self.wh_head(feat)
        center2kpt_offset_pred = self.center2kpt_offset_head(feat)
        dim_pred = self.dim_head(feat)
        depth_pred = self.depth_head(feat)
        depth_pred[:, 0, :, :] = 1. / (depth_pred[:, 0, :, :].sigmoid() + EPS) - 1

        alpha_feat = self.dir_feat(feat)
        alpha_cls_pred = self.dir_cls(alpha_feat)
        alpha_offset_pred = self.dir_reg(alpha_feat)
        return center_heatmap_pred, wh_pred, offset_pred, center2kpt_offset_pred, kpt_heatmap_pred, \
               kpt_heatmap_offset_pred, dim_pred, alpha_cls_pred, alpha_offset_pred, depth_pred

    @staticmethod
    def extract_input_from_tensor(input, ind, mask):
        input = transpose_and_gather_feat(input, ind)
        return input[mask]

    @staticmethod
    def extract_target_from_tensor(target, mask):
        return target[mask]

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class and residual. '''
        angle = angle % (2 * PI)
        assert (angle >= 0 and angle <= 2 * PI)
        angle_per_class = 2 * PI / float(self.num_alpha_bins)
        shifted_angle = (angle + angle_per_class / 2) % (2 * PI)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, cls, residual):
        ''' Inverse function to angle2class. '''
        angle_per_class = 2 * PI / float(self.num_alpha_bins)
        angle_center = cls * angle_per_class
        angle = angle_center + residual
        return angle

    def decode_alpha_multibin(self, alpha_cls, alpha_offset):
        alpha_score, cls = alpha_cls.max(dim=-1)
        cls = cls.unsqueeze(2)
        alpha_offset = alpha_offset.gather(2, cls)
        alpha = self.class2angle(cls, alpha_offset)

        alpha[alpha > PI] = alpha[alpha > PI] - 2 * PI
        alpha[alpha < -PI] = alpha[alpha < -PI] + 2 * PI
        return alpha

    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   center2kpt_offset_preds,
                   kpt_heatmap_preds,
                   kpt_heatmap_offset_preds,
                   dim_preds,
                   alpha_cls_preds,
                   alpha_offset_preds,
                   depth_preds,
                   img_metas,
                   rescale=False):
        P2 = [img_meta['cam2img'] for img_meta in img_metas]

        result_list = self.decode_heatmap(
            center_heatmap_preds[0],
            wh_preds[0],
            offset_preds[0],
            center2kpt_offset_preds[0],
            kpt_heatmap_preds[0],
            kpt_heatmap_offset_preds[0],
            dim_preds[0],
            alpha_cls_preds[0],
            alpha_offset_preds[0],
            depth_preds[0],
            img_metas[0]['pad_shape'][:2],
            P2,
            cfg=self.test_cfg)

        return result_list

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       center2kpt_offset_pred,
                       kpt_heatmap_pred,
                       kpt_heatmap_offset_pred,
                       dim_pred,
                       alpha_cls_pred,
                       alpha_offset_pred,
                       depth_pred,
                       img_shape,
                       camera_intrinsics,
                       cfg=None,
                       ):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        assert cfg is not None
        thresh = cfg.thresh
        top_k = cfg.topk

        batch, cat, height, width = center_heatmap_pred.shape
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=cfg.local_maximum_kernel)

        res = []
        for i in range(batch):
            camera_intrinsic = camera_intrinsics[i]

            *batch_dets, ys, xs = get_topk_from_heatmap(
                center_heatmap_pred[i:i + 1], k=top_k)
            batch_scores, batch_index, batch_topk_labels = batch_dets

            # decode 3D prediction
            dim = transpose_and_gather_feat(dim_pred[i:i + 1], batch_index)
            alpha_cls = transpose_and_gather_feat(alpha_cls_pred[i:i + 1], batch_index)
            alpha_offset = transpose_and_gather_feat(alpha_offset_pred[i:i + 1], batch_index)
            depth_pred_single = transpose_and_gather_feat(depth_pred[i:i + 1], batch_index)
            depth = depth_pred_single[:, :, 0:1]

            # change predict score based on sigma
            sigma = depth_pred_single[:, :, 1]
            sigma = torch.exp(-sigma)

            batch_scores *= sigma

            # 0. get kpts prediction
            center2kpt_offset = transpose_and_gather_feat(center2kpt_offset_pred[i:i + 1], batch_index)
            center2kpt_offset = center2kpt_offset.view(1, top_k, self.num_kpt * 2)[..., -2:]
            center2kpt_offset[..., ::2] += xs.view(1, top_k, 1).expand(1, top_k, 1)
            center2kpt_offset[..., 1::2] += ys.view(1, top_k, 1).expand(1, top_k, 1)

            kpts = center2kpt_offset

            kpts[..., ::2] *= (inp_w / width)
            kpts[..., 1::2] *= (inp_h / height)

            # 1. decode alpha
            alpha = self.decode_alpha_multibin(alpha_cls, alpha_offset)  # (b, k, 1)

            # 1.5 get projected center
            center2d = kpts  # (b, k, 2)

            # 2. recover rotY
            rot_y = self.recover_rotation(kpts, alpha, camera_intrinsic)  # (b, k, 3)

            # 2.5 recover box3d_center from center2d and depth
            center3d = torch.cat([center2d, depth], dim=-1).squeeze(0)
            center3d = self.pts2Dto3D(center3d, np.array(camera_intrinsic)).unsqueeze(0)

            # 3. compose 3D box
            batch_bboxes_3d = torch.cat([center3d, dim, rot_y], dim=-1).squeeze(0)
            batch_topk_labels = batch_topk_labels.squeeze(0)
            batch_scores = batch_scores.squeeze(0)

            mask = batch_scores > thresh
            batch_scores = batch_scores[mask]
            batch_bboxes_3d = batch_bboxes_3d[mask]
            batch_topk_labels = batch_topk_labels[mask]

            res.append((batch_bboxes_3d, batch_topk_labels, batch_scores))

        return res

    def recover_rotation(self, kpts, alpha, calib):
        device = kpts.device
        calib = torch.tensor(calib).type(torch.FloatTensor).to(device).unsqueeze(0)

        si = torch.zeros_like(kpts[:, :, 0:1]) + calib[:, 0:1, 0:1]
        rot_y = alpha + torch.atan2(kpts[:, :, 0:1] - calib[:, 0:1, 2:3], si)

        while (rot_y > PI).any():
            rot_y[rot_y > PI] = rot_y[rot_y > PI] - 2 * PI
        while (rot_y < -PI).any():
            rot_y[rot_y < -PI] = rot_y[rot_y < -PI] + 2 * PI

        return rot_y

    @staticmethod
    def pts2Dto3D(points, view):
        """
        Args:
            points (torch.Tensor): points in 2D images, [N, 3], \
                3 corresponds with x, y in the image and depth.
            view (np.ndarray): camera instrinsic, [3, 3]

        Returns:
            torch.Tensor: points in 3D space. [N, 3], \
                3 corresponds with x, y, z in 3D space.
        """
        assert view.shape[0] <= 4
        assert view.shape[1] <= 4
        assert points.shape[1] == 3

        points2D = points[:, :2]
        depths = points[:, 2].view(-1, 1)
        unnorm_points2D = torch.cat([points2D * depths, depths], dim=1)

        viewpad = torch.eye(4, dtype=points2D.dtype, device=points2D.device)
        viewpad[:view.shape[0], :view.shape[1]] = points2D.new_tensor(view)
        inv_viewpad = torch.inverse(viewpad).transpose(0, 1)

        # Do operation in homogenous coordinates.
        nbr_points = unnorm_points2D.shape[0]
        homo_points2D = torch.cat(
            [unnorm_points2D,
             points2D.new_ones((nbr_points, 1))], dim=1)
        points3D = torch.mm(homo_points2D, inv_viewpad)[:, :3]

        return points3D

    @staticmethod
    def _topk_channel(scores, K=40):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        return topk_scores, topk_inds, topk_ys, topk_xs

    def forward_train(self,
                      x,
                      img_metas,
                      **kwargs):
        outs = self(x)
        proposals_list = self.get_bboxes(*outs, img_metas)

        result = dict(proposals_list=proposals_list)
        
        return result

    def forward_test(self,
                     x,
                     img_metas,
                     rescale=False):

        outs = self(x)
        proposal_list = self.get_bboxes(*outs, img_metas)
        return proposal_list
