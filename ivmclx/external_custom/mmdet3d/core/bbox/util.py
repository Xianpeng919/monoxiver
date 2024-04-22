'''modified from DETR3D https://github.com/WangYueFt/detr3d/blob/main/projects/mmdet3d_plugin/models/dense_heads/detr3d_head.py'''
import torch 


def normalize_bbox(bboxes, pc_range=None):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    # l = bboxes[..., 3:4].log()
    # h = bboxes[..., 4:5].log()
    # w = bboxes[..., 5:6].log()
    l = bboxes[..., 3:4]
    h = bboxes[..., 4:5]
    w = bboxes[..., 5:6]

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8] 
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, cz, l, h, w, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, cz, l, h, w, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes

def denormalize_bbox(normalized_bboxes, pc_range=None):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 2:3]

    # size
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 4:5]
    w = normalized_bboxes[..., 5:6]

    # l = l.exp()
    # h = h.exp()
    # w = w.exp()
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, l, h, w, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, l, h, w, rot], dim=-1)
    return denormalized_bboxes