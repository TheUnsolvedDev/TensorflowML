import tensorflow as tf
from utils import *
from config import IMAGE_SIZE, NUM_CLASSES


def build_rcnn_model(
    num_classes=NUM_CLASSES,
    anchor_scales=[32, 64, 128],
    anchor_ratios=[0.5, 1.0, 2.0],
    feature_stride=16,
    roi_size=(7, 7),
    top_n=64
):
    # Input
    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image_input")

    # Backbone
    resnet = tf.keras.applications.ResNet50(
        include_top=False, input_tensor=inputs)
    feature_map = resnet.get_layer("conv4_block6_out").output  # [B, H, W, C]

    # RPN Head
    rpn_conv = tf.keras.layers.Conv2D(
        256, 3, padding='same', activation='relu')(feature_map)

    num_anchors = len(anchor_scales) * len(anchor_ratios)
    rpn_cls_logits = tf.keras.layers.Conv2D(
        num_anchors * 1, 1, name="rpn_cls_logits")(rpn_conv)
    rpn_bbox_deltas = tf.keras.layers.Conv2D(
        num_anchors * 4, 1, name="rpn_bbox_deltas")(rpn_conv)

    # Generate anchors (constant, based on feature map size)
    feat_h, feat_w = IMAGE_SIZE // feature_stride, IMAGE_SIZE // feature_stride
    anchors = generate_anchors(
        anchor_scales, anchor_ratios, feat_h, feature_stride)

    # Proposal layer logic
    def proposal_layer(args):
        rpn_cls, rpn_reg = args
        B = tf.shape(rpn_cls)[0]
        A = anchors.shape[0]

        rpn_cls_flat = tf.reshape(rpn_cls, [B, -1])
        rpn_reg_flat = tf.reshape(rpn_reg, [B, A, 4])

        proposals_list = []
        indices_list = []

        for i in tf.range(B):
            scores = tf.nn.sigmoid(rpn_cls_flat[i])
            deltas = rpn_reg_flat[i]
            decoded = decode_boxes(anchors, deltas)
            clipped = tf.clip_by_value(decoded, 0.0, 1.0)
            selected = filter_proposals(clipped, scores, max_proposals=top_n)
            proposals_list.append(selected)
            indices_list.append(
                tf.ones((tf.shape(selected)[0],), dtype=tf.int32) * i)

        return tf.concat(proposals_list, axis=0), tf.concat(indices_list, axis=0)

    proposal_layer = ProposalLayer(anchors, top_n)
    proposals, proposal_indices = proposal_layer([rpn_cls_logits, rpn_bbox_deltas])


    # ROIAlign
    roi_align_layer = ROIAlign(output_size=roi_size)
    pooled_rois = roi_align_layer(feature_map, proposals, proposal_indices)
    pooled_features = tf.keras.layers.GlobalAveragePooling2D()(pooled_rois)

    # RCNN Head
    fc1 = tf.keras.layers.Dense(1024, activation='relu')(pooled_features)
    fc2 = tf.keras.layers.Dense(1024, activation='relu')(fc1)
    class_logits = tf.keras.layers.Dense(num_classes, name="class_logits")(fc2)
    bbox_preds = tf.keras.layers.Dense(num_classes * 4, name="bbox_preds")(fc2)

    # Final model
    model = tf.keras.Model(inputs=inputs,
                           outputs=[class_logits, bbox_preds,
                                    rpn_cls_logits, rpn_bbox_deltas],
                           name="Functional_RCNN")
    model.summary()
    return model


class RPNClassificationLoss(tf.keras.losses.Loss):
    def __init__(self, name="rpn_class_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        """
        y_true: [N] in {-1, 0, 1} where -1 = ignore, 0 = negative, 1 = positive
        y_pred: [N] logits (not probabilities)
        """
        # Only keep where label != -1
        valid_mask = tf.not_equal(y_true, -1)
        y_true = tf.boolean_mask(y_true, valid_mask)
        y_pred = tf.boolean_mask(y_pred, valid_mask)
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y_true, tf.float32), logits=y_pred))


class RPNRegressionLoss(tf.keras.losses.Loss):
    def __init__(self, name="rpn_box_loss", delta=1.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = delta

    def call(self, y_true, y_pred):
        """
        y_true: [N, 4], valid targets for positive anchors
        y_pred: [N, 4], predicted deltas for the same anchors
        """
        loss = tf.reduce_sum(tf.where(
            tf.abs(y_true - y_pred) < self.delta,
            0.5 * tf.square(y_true - y_pred),
            self.delta * (tf.abs(y_true - y_pred) - 0.5 * self.delta)
        ), axis=-1)
        return tf.reduce_mean(loss)


class RCNNClassificationLoss(tf.keras.losses.Loss):
    def __init__(self, name="rcnn_class_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        """
        y_true: [N] class indices
        y_pred: [N, num_classes] logits
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))


class RCNNRegressionLoss(tf.keras.losses.Loss):
    def __init__(self, name="rcnn_box_loss", num_classes=91, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        """
        y_true: [N, 4]
        y_pred: [N, num_classes * 4]
        NOTE: y_pred must be reshaped to [N, num_classes, 4] and select based on true class.
        """
        y_true_boxes, y_true_labels = y_true

        y_pred = tf.reshape(y_pred, [-1, self.num_classes, 4])
        batch_indices = tf.range(tf.shape(y_true_labels)[0])
        gathered = tf.gather_nd(y_pred, tf.stack(
            [batch_indices, y_true_labels], axis=1))  # [N, 4]

        return tf.reduce_mean(tf.reduce_sum(tf.square(y_true_boxes - gathered), axis=-1))
    
if __name__ == '__main__':
    model = build_rcnn_model()
