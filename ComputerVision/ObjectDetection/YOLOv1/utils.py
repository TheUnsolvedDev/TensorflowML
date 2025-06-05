import tensorflow as tf

def compute_iou(boxes1, boxes2):
    # boxes1, boxes2: (batch, S, S, B, 4) or (batch, S, S, 1, 4)
    boxes1_xy = boxes1[..., :2] - boxes1[..., 2:] * 0.5
    boxes1_wh = boxes1[..., :2] + boxes1[..., 2:] * 0.5
    boxes2_xy = boxes2[..., :2] - boxes2[..., 2:] * 0.5
    boxes2_wh = boxes2[..., :2] + boxes2[..., 2:] * 0.5

    intersect_mins = tf.maximum(boxes1_xy, boxes2_xy)
    intersect_maxs = tf.minimum(boxes1_wh, boxes2_wh)
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    boxes1_area = (boxes1_wh[..., 0] - boxes1_xy[..., 0]) * (boxes1_wh[..., 1] - boxes1_xy[..., 1])
    boxes2_area = (boxes2_wh[..., 0] - boxes2_xy[..., 0]) * (boxes2_wh[..., 1] - boxes2_xy[..., 1])

    union_area = boxes1_area + boxes2_area - intersect_area
    return intersect_area / tf.maximum(union_area, 1e-6)
class CoordLossMetric(tf.keras.metrics.Metric):
    def __init__(self, name="coord_loss", lambda_coord=5.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_coord = lambda_coord
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        object_mask = y_true[..., 4:5]
        pred_box = y_pred[..., 0:4]
        true_box = y_true[..., 0:4]
        pred_wh = tf.maximum(pred_box[..., 2:4], 1e-6)
        true_wh = tf.maximum(true_box[..., 2:4], 1e-6)
        coord_xy_loss = tf.reduce_mean(
            object_mask * tf.square(pred_box[..., 0:2] - true_box[..., 0:2]))
        coord_wh_loss = tf.reduce_mean(
            object_mask * tf.square(tf.sqrt(pred_wh) - tf.sqrt(true_wh)))
        loss = self.lambda_coord * (coord_xy_loss + coord_wh_loss)
        self.total.assign_add(loss)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class ObjectnessLossMetric(tf.keras.metrics.Metric):
    def __init__(self, name="obj_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        object_mask = y_true[..., 4:5]
        pred_conf = y_pred[..., 4:5]
        true_conf = y_true[..., 4:5]
        loss = tf.reduce_mean(object_mask * tf.square(pred_conf - true_conf))
        self.total.assign_add(loss)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class NoObjectnessLossMetric(tf.keras.metrics.Metric):
    def __init__(self, lambda_noobj=0.5, name="noobj_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_noobj = lambda_noobj
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        object_mask = y_true[..., 4:5]
        noobj_mask = 1.0 - object_mask
        pred_conf = y_pred[..., 4:5]
        true_conf = y_true[..., 4:5]
        loss = self.lambda_noobj * \
            tf.reduce_mean(noobj_mask * tf.square(pred_conf - true_conf))
        self.total.assign_add(loss)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class ClassLossMetric(tf.keras.metrics.Metric):
    def __init__(self, name="class_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        object_mask = y_true[..., 4:5]
        pred_class = y_pred[..., 5:]
        true_class = y_true[..., 5:]
        ce = tf.keras.losses.categorical_crossentropy(
            true_class, pred_class, from_logits=True)
        loss = tf.reduce_mean(tf.squeeze(object_mask, axis=-1) * ce)
        self.total.assign_add(loss)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)
        
class ClassAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="class_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        object_mask = y_true[..., 4:5]
        pred_class = tf.argmax(y_pred[..., 5:], axis=-1)
        true_class = tf.argmax(y_true[..., 5:], axis=-1)
        match = tf.cast(tf.equal(pred_class, true_class), tf.float32)
        match = tf.squeeze(object_mask, axis=-1) * match
        self.correct.assign_add(tf.reduce_sum(match))
        self.total.assign_add(tf.reduce_sum(tf.squeeze(object_mask, axis=-1)))

    def result(self):
        return self.correct / (self.total + 1e-6)

    def reset_states(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)
