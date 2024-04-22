import torch
import torch.nn as nn

from mmdet.models.builder import HEADS
from mmdet3d.core import points_cam2img

from ivmclx.external_custom.mmdet3d.core.bbox.util import normalize_bbox


@HEADS.register_module()
class MonoXiverProposalGenerator(nn.Module):
    def __init__(self,
                 num_class=3,
                 x_stride=0.75,
                 x_max=1.5,
                 z_stride=0.75,
                 z_max=1.5):
        super(MonoXiverProposalGenerator, self).__init__()
        self.num_class = num_class

        self.x_grid, self.z_grid, self.num_augments = self.generate_grid(x_max,
                                                                        x_stride,
                                                                        z_max,
                                                                        z_stride)

        self.x_grid = self.x_grid.cuda()
        self.z_grid = self.z_grid.cuda()

    @staticmethod
    def generate_grid(x_max, x_stride, z_max, z_stride):
        x_range = torch.arange(-x_max, x_max + x_stride, x_stride)
        z_range = torch.arange(-z_max, z_max + z_stride, z_stride)

        z_grid, x_grid = torch.meshgrid(z_range, x_range)
        z_grid = z_grid.flatten()
        x_grid = x_grid.flatten()
        num_augments = len(z_grid)

        return x_grid, z_grid, num_augments

    def get_projections(self, box3d, box_type_3d, P2):
        box3d_cam = box_type_3d(box3d, box_dim=box3d.shape[-1], origin=(0.5, 0.5, 0.5))

        box3d_center = box3d[..., :3].unsqueeze(1)
        box_corners = box3d_cam.corners
        box_corners = torch.cat([box_corners, box3d_center], dim=1)

        box_corners_in_image = points_cam2img(box_corners, P2, with_depth=True)

        corners_depth = box_corners_in_image[..., 2]
        box_corners_in_image = box_corners_in_image[..., :2]
        # box_corners_in_image: [N, 18]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]

        box_2d_pred = torch.cat([minxy, maxxy], dim=1)
        projected_corners = box_corners_in_image.reshape(-1, 18)

        return box_2d_pred, projected_corners, corners_depth

    def generate_box_based_on_grid(self,
                                   bbox_3d_pred,
                                   label_pred,
                                   score_pred,
                                   P2,
                                   x_grid,
                                   z_grid,
                                   num_augments,
                                   box_type_3d):
        new_box3d = bbox_3d_pred.unsqueeze(0).repeat(num_augments, 1)
        new_box3d[..., 0] += x_grid
        new_box3d[..., 2] += z_grid

        new_labels = label_pred.repeat(num_augments)
        new_scores = score_pred.repeat(num_augments)
        new_scores_logits = new_scores.new_zeros(len(new_scores), self.num_class)
        new_scores_logits[:, new_labels] = 1.

        new_box_2d, new_projected_corners, new_corners_depth = self.get_projections(
            new_box3d, box_type_3d, P2)

        return new_box3d, new_labels, new_scores, new_box_2d, new_projected_corners, \
                new_corners_depth, new_scores_logits

    def generate_proposals(self,
                           bboxes_3d_pred,
                           labels_pred,
                           scores_pred,
                           img_meta):
        # generate proposals per image
        box_type_3d = img_meta['box_type_3d']
        P2 = img_meta['cam2img']
        P2 = bboxes_3d_pred.new_tensor(P2)

        proposal_bboxes3d_augment = []
        proposal_bboxes3d_normalized_augment = []
        proposal_bboxes2d_augment = []
        proposal_projected_corners_augment = []
        proposal_corners_depth_augment = []
        proposal_labels_augment = []
        proposal_scores_augment = []
        proposal_scores_logits_augment = []

        proposal_bboxes3d = []
        proposal_bboxes3d_normalized = []
        proposal_bboxes2d = []
        proposal_projected_corners = []
        proposal_corners_depth = []
        proposal_labels = []
        proposal_scores = []
        proposal_scores_logits = []

        if len(bboxes_3d_pred) > 0:
            # TODO: generate boxes without iteration
            for bbox_3d_pred, label_pred, score_pred in zip(bboxes_3d_pred,
                                                            labels_pred,
                                                            scores_pred):

                new_box3d, new_labels, new_scores, new_box_2d, new_projected_corners, \
                    new_corners_depth, new_scores_logits \
                    = self.generate_box_based_on_grid(bbox_3d_pred,
                                                    label_pred,
                                                    score_pred,
                                                    P2,
                                                    self.x_grid,
                                                    self.z_grid,
                                                    self.num_augments,
                                                    box_type_3d)

                proposal_bboxes3d_augment.append(new_box3d)
                proposal_bboxes3d_normalized_augment.append(normalize_bbox(new_box3d))
                proposal_bboxes2d_augment.append(new_box_2d)
                proposal_projected_corners_augment.append(new_projected_corners)
                proposal_corners_depth_augment.append(new_corners_depth)
                proposal_labels_augment.append(new_labels)
                proposal_scores_augment.append(new_scores)
                proposal_scores_logits_augment.append(new_scores_logits)

        if len(proposal_labels) > 0:
            proposal_bboxes3d = torch.cat(proposal_bboxes3d, dim=0)
            proposal_bboxes3d_normalized = torch.cat(proposal_bboxes3d_normalized, dim=0)
            proposal_bboxes2d = torch.cat(proposal_bboxes2d, dim=0)
            proposal_projected_corners = torch.cat(proposal_projected_corners, dim=0)
            proposal_corners_depth = torch.cat(proposal_corners_depth, dim=0)
            proposal_labels = torch.cat(proposal_labels, dim=0)
            proposal_scores = torch.cat(proposal_scores, dim=0)
            proposal_scores_logits = torch.cat(proposal_scores_logits, dim=0)
        else:
            proposal_bboxes3d = torch.zeros(0, 7).cuda()
            proposal_bboxes3d_normalized = torch.zeros(0, 8).cuda()
            proposal_bboxes2d = torch.zeros(0, 4).cuda()
            proposal_projected_corners = torch.zeros(0, 18).cuda()
            proposal_corners_depth = torch.zeros(0, 9).cuda()
            proposal_labels = torch.zeros(0).cuda()
            proposal_scores = torch.zeros(0).cuda()
            proposal_scores_logits = torch.zeros(0, self.num_class).cuda()
        
        if len(proposal_labels_augment) > 0:
            proposal_bboxes3d_augment = torch.cat(proposal_bboxes3d_augment, dim=0)
            proposal_bboxes3d_normalized_augment = torch.cat(proposal_bboxes3d_normalized_augment, dim=0)
            proposal_bboxes2d_augment = torch.cat(proposal_bboxes2d_augment, dim=0)
            proposal_projected_corners_augment = torch.cat(proposal_projected_corners_augment, dim=0)
            proposal_corners_depth_augment = torch.cat(proposal_corners_depth_augment, dim=0)
            proposal_labels_augment = torch.cat(proposal_labels_augment, dim=0)
            proposal_scores_augment = torch.cat(proposal_scores_augment, dim=0)
            proposal_scores_logits_augment = torch.cat(proposal_scores_logits_augment, dim=0)
        else:
            proposal_bboxes3d_augment = torch.zeros(0, 7).cuda()
            proposal_bboxes3d_normalized_augment = torch.zeros(0, 8).cuda()
            proposal_bboxes2d_augment = torch.zeros(0, 4).cuda()
            proposal_projected_corners_augment = torch.zeros(0, 18).cuda()
            proposal_corners_depth_augment = torch.zeros(0, 9).cuda()
            proposal_labels_augment = torch.zeros(0).cuda()
            proposal_scores_augment = torch.zeros(0).cuda()
            proposal_scores_logits_augment = torch.zeros(0, self.num_class).cuda()
        
        raw_set = [proposal_bboxes3d, proposal_bboxes3d_normalized, proposal_bboxes2d, proposal_projected_corners, \
                proposal_corners_depth, proposal_labels, proposal_scores, proposal_scores_logits]
        augment_set = [proposal_bboxes3d_augment, proposal_bboxes3d_normalized_augment, proposal_bboxes2d_augment, proposal_projected_corners_augment, \
                proposal_corners_depth_augment, proposal_labels_augment, proposal_scores_augment, proposal_scores_logits_augment]

        return raw_set, augment_set

    def forward(self, initial_proposals_list, img_metas):
        assert len(img_metas) == len(initial_proposals_list)

        proposals_bboxes_3d = []
        proposals_bboxes_3d_normalized = []
        proposals_bboxes_2d = []
        proposals_projected_corners = []
        proposals_corners_depth = []
        proposals_labels = []
        proposals_scores = []
        proposals_scores_logits = []

        # [30, 8], [30], [30]
        for initial_proposal_list, img_meta in zip(initial_proposals_list,
                                                   img_metas):
            initial_bboxes_3d_pred, initial_labels_pred, initial_scores_pred = initial_proposal_list[:3]
            assert len(initial_bboxes_3d_pred) == len(initial_labels_pred) == len(initial_scores_pred)
            raw_set, augment_set = self.generate_proposals(initial_bboxes_3d_pred.detach(),
                                                            initial_labels_pred.detach(),
                                                            initial_scores_pred.detach(),
                                                            img_meta)

            # proposal_bboxes2d: N, 4
            # proposal_bboxes_3d: N, 7 (Tensor)
            # proposal_bboxes_3d_normalized: N, 8 (Tensor)
            # proposal_projected_corners: N, 18
            # proposal_corners_depth: N, 9
            # proposal_labels: N
            # proposal_scores: N
            proposal_bboxes_3d, proposal_bboxes_3d_normalized, proposal_bboxes_2d, proposal_projected_corners, proposal_corners_depth, \
                proposal_labels, proposal_scores, proposal_scores_logits = raw_set
            proposal_bboxes_3d_augment, proposal_bboxes_3d_normalized_augment, proposal_bboxes_2d_augment, \
            proposal_projected_corners_augment, proposal_corners_depth_augment, \
                proposal_labels_augment, proposal_scores_augment, proposal_scores_logits_augment = augment_set

            proposals_bboxes_3d.append([proposal_bboxes_3d, proposal_bboxes_3d_augment])
            proposals_bboxes_3d_normalized.append([proposal_bboxes_3d_normalized, proposal_bboxes_3d_normalized_augment])
            proposals_bboxes_2d.append([proposal_bboxes_2d, proposal_bboxes_2d_augment])
            proposals_projected_corners.append([proposal_projected_corners, proposal_projected_corners_augment])
            proposals_corners_depth.append([proposal_corners_depth, proposal_corners_depth_augment])
            proposals_labels.append([proposal_labels, proposal_labels_augment])
            proposals_scores.append([proposal_scores, proposal_scores_augment])
            proposals_scores_logits.append([proposal_scores_logits, proposal_scores_logits_augment])

        proposals_dict = dict(
            proposals_bboxes_3d=proposals_bboxes_3d,
            proposals_bboxes_3d_normalized=proposals_bboxes_3d_normalized,
            proposals_bboxes_2d=proposals_bboxes_2d,
            proposals_projected_corners=proposals_projected_corners,
            proposals_corners_depth=proposals_corners_depth,
            proposals_labels=proposals_labels,
            proposals_scores=proposals_scores,
            proposals_scores_logits=proposals_scores_logits,
        )
        return proposals_dict
