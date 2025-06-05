import tensorflow as tf
import numpy as np
from config import IMAGE_SIZE


def generate_anchors(scales, ratios, feature_size, stride):
    anchors = []
    for y in range(feature_size):
        for x in range(feature_size):
            cx = x * stride + stride / 2
            cy = y * stride + stride / 2
            for scale in scales:
                for ratio in ratios:
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)
                    xmin = cx - w / 2
                    ymin = cy - h / 2
                    xmax = cx + w / 2
                    ymax = cy + h / 2
                    anchors.append([ymin, xmin, ymax, xmax])
    return tf.convert_to_tensor(anchors, dtype=tf.float32) / IMAGE_SIZE


def compute_iou(boxes1, boxes2):
    """
    boxes1: [N, 4]  (anchors)
    boxes2: [M, 4]  (gt boxes)
    returns: [N, M] iou matrix
    """
    boxes1 = tf.expand_dims(boxes1, 1)  # [N, 1, 4]
    boxes2 = tf.expand_dims(boxes2, 0)  # [1, M, 4]

    ymin1, xmin1, ymax1, xmax1 = tf.split(boxes1, 4, axis=-1)
    ymin2, xmin2, ymax2, xmax2 = tf.split(boxes2, 4, axis=-1)

    inter_ymin = tf.maximum(ymin1, ymin2)
    inter_xmin = tf.maximum(xmin1, xmin2)
    inter_ymax = tf.minimum(ymax1, ymax2)
    inter_xmax = tf.minimum(xmax1, xmax2)

    inter_area = tf.maximum(inter_ymax - inter_ymin, 0) * \
        tf.maximum(inter_xmax - inter_xmin, 0)

    area1 = (ymax1 - ymin1) * (xmax1 - xmin1)
    area2 = (ymax2 - ymin2) * (xmax2 - xmin2)

    union = area1 + area2 - inter_area
    return inter_area / tf.maximum(union, 1e-6)


def match_anchors_to_gt(anchors, gt_boxes, iou_pos_thresh=0.7, iou_neg_thresh=0.3):
    ious = compute_iou(anchors, gt_boxes)
    max_iou = tf.reduce_max(ious, axis=1)
    max_indices = tf.argmax(ious, axis=1)

    labels = tf.where(max_iou >= iou_pos_thresh, 1, 0)
    labels = tf.where(max_iou < iou_neg_thresh, -1, labels)

    matched_gt = tf.gather(gt_boxes, max_indices)
    return labels, matched_gt


class ROIAlign(tf.keras.layers.Layer):
    def __init__(self, output_size=(7, 7)):
        super().__init__()
        self.output_size = output_size

    def call(self, feature_map, boxes, box_indices):
        return tf.image.crop_and_resize(
            image=feature_map,
            boxes=boxes,
            box_indices=box_indices,  # ✅ correct keyword here
            crop_size=self.output_size,
            method='bilinear'
        )


def decode_boxes(anchors, deltas):
    """
    Apply predicted deltas to anchor boxes to get proposals.
    anchors, deltas: [N, 4] in normalized (ymin, xmin, ymax, xmax)
    """
    # Convert to center format
    ymin, xmin, ymax, xmax = tf.split(anchors, 4, axis=-1)
    anchor_h = ymax - ymin
    anchor_w = xmax - xmin
    anchor_cy = ymin + 0.5 * anchor_h
    anchor_cx = xmin + 0.5 * anchor_w

    dy, dx, dh, dw = tf.split(deltas, 4, axis=-1)
    pred_cy = dy * anchor_h + anchor_cy
    pred_cx = dx * anchor_w + anchor_cx
    pred_h = tf.exp(dh) * anchor_h
    pred_w = tf.exp(dw) * anchor_w

    pred_ymin = pred_cy - 0.5 * pred_h
    pred_xmin = pred_cx - 0.5 * pred_w
    pred_ymax = pred_cy + 0.5 * pred_h
    pred_xmax = pred_cx + 0.5 * pred_w

    return tf.concat([pred_ymin, pred_xmin, pred_ymax, pred_xmax], axis=-1)


def filter_proposals(proposals, scores, max_proposals=200, iou_threshold=0.7):
    """
    proposals: [N, 4]
    scores: [N]
    Returns top-N proposals after NMS
    """
    selected_indices = tf.image.non_max_suppression(
        proposals,
        scores,
        max_output_size=max_proposals,
        iou_threshold=iou_threshold
    )
    return tf.gather(proposals, selected_indices)


def encode_boxes(gt_boxes, anchors):
    """
    Compute deltas (dy, dx, dh, dw) from anchors to gt_boxes
    gt_boxes, anchors: [N, 4] in [ymin, xmin, ymax, xmax]
    """
    ymin, xmin, ymax, xmax = tf.split(gt_boxes, 4, axis=-1)
    anchor_ymin, anchor_xmin, anchor_ymax, anchor_xmax = tf.split(
        anchors, 4, axis=-1)

    gt_h = ymax - ymin
    gt_w = xmax - xmin
    gt_cy = ymin + 0.5 * gt_h
    gt_cx = xmin + 0.5 * gt_w

    anchor_h = anchor_ymax - anchor_ymin
    anchor_w = anchor_xmax - anchor_xmin
    anchor_cy = anchor_ymin + 0.5 * anchor_h
    anchor_cx = anchor_xmin + 0.5 * anchor_w

    dy = (gt_cy - anchor_cy) / anchor_h
    dx = (gt_cx - anchor_cx) / anchor_w
    dh = tf.math.log(gt_h / anchor_h)
    dw = tf.math.log(gt_w / anchor_w)

    return tf.concat([dy, dx, dh, dw], axis=-1)


def rpn_target_assign(anchors, gt_boxes, iou_pos_thresh=0.7, iou_neg_thresh=0.3):
    labels, matched_gt = match_anchors_to_gt(
        anchors, gt_boxes, iou_pos_thresh, iou_neg_thresh)
    bbox_targets = encode_boxes(matched_gt, anchors)
    return labels, bbox_targets


def proposal_target_assign(proposals, gt_boxes, gt_labels, num_samples=64, fg_fraction=0.25, pos_iou_thresh=0.5, neg_iou_thresh=0.1):
    """
    Assign proposals to GT boxes for training the RCNN head.
    Returns: sampled_proposals, matched_gt_boxes, matched_labels
    """
    ious = compute_iou(proposals, gt_boxes)
    max_iou = tf.reduce_max(ious, axis=1)
    matched_idx = tf.argmax(ious, axis=1)

    labels = tf.gather(gt_labels, matched_idx)
    labels = tf.where(max_iou >= pos_iou_thresh, labels, -1)
    labels = tf.where(max_iou < neg_iou_thresh, 0, labels)

    # Sample fixed number of rois
    pos_idx = tf.where(labels > 0)[:, 0]
    neg_idx = tf.where(labels == 0)[:, 0]

    num_fg = int(num_samples * fg_fraction)
    num_fg = tf.minimum(tf.size(pos_idx), num_fg)
    num_bg = num_samples - num_fg
    num_bg = tf.minimum(tf.size(neg_idx), num_bg)

    pos_idx = tf.random.shuffle(pos_idx)[:num_fg]
    neg_idx = tf.random.shuffle(neg_idx)[:num_bg]

    sample_idx = tf.concat([pos_idx, neg_idx], axis=0)

    sampled_proposals = tf.gather(proposals, sample_idx)
    matched_gt_boxes = tf.gather(gt_boxes, tf.gather(matched_idx, sample_idx))
    matched_labels = tf.gather(labels, sample_idx)

    # Fallback in case no valid proposals are found
    if tf.shape(sampled_proposals)[0] == 0:
        sampled_proposals = tf.expand_dims(proposals[0], axis=0)
        matched_gt_boxes = tf.expand_dims(gt_boxes[0], axis=0)
        matched_labels = tf.expand_dims(gt_labels[0], axis=0)

    # Ensure clipping for safety
    sampled_proposals = tf.clip_by_value(sampled_proposals, 0.0, 1.0)
    matched_gt_boxes = tf.clip_by_value(matched_gt_boxes, 0.0, 1.0)

    return sampled_proposals, matched_gt_boxes, matched_labels

class ProposalLayer(tf.keras.layers.Layer):
    def __init__(self, anchors, top_n=64, **kwargs):
        super().__init__(**kwargs)
        self.anchors = anchors
        self.top_n = top_n

    def call(self, inputs):
        rpn_cls, rpn_reg = inputs
        B = tf.shape(rpn_cls)[0]
        A = tf.shape(self.anchors)[0]

        def process_single_sample(args):
            cls_i, reg_i = args
            scores = tf.reshape(tf.nn.sigmoid(cls_i), [-1])
            deltas = tf.reshape(reg_i, [A, 4])
            decoded = decode_boxes(self.anchors, deltas)
            clipped = tf.clip_by_value(decoded, 0.0, 1.0)
            selected = filter_proposals(clipped, scores, max_proposals=self.top_n)

            batch_inds = tf.ones(tf.shape(selected)[0], dtype=tf.int32) * 0  # filled during map_fn
            return selected, batch_inds

        proposals, indices = tf.map_fn(
            process_single_sample,
            (rpn_cls, rpn_reg),
            fn_output_signature=(
                tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            )
        )

        # Flatten across batch
        proposals = tf.reshape(proposals, [-1, 4])
        indices = tf.reshape(indices, [-1])

        return proposals, indices
