import tensorflow as tf
import json
import os
import numpy as np
from config import *
from collections import defaultdict

class COCORCNNLoader:
    def __init__(self, coco_dir="../coco", batch_size=BATCH_SIZE):
        self.coco_dir = coco_dir
        self.batch_size = batch_size

    def parse_coco_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
        image_to_anns = defaultdict(list)

        for ann in data["annotations"]:
            image_id = ann["image_id"]
            bbox = ann["bbox"]  # COCO: [x, y, width, height]
            category_id = ann["category_id"]
            image_to_anns[image_id].append((bbox, category_id))

        return id_to_file, image_to_anns

    def load_and_preprocess_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        return tf.cast(image, tf.float32) / 255.0

    def convert_to_yxyx(self, bboxes):
        # COCO: [x, y, w, h] → [ymin, xmin, ymax, xmax]
        converted = []
        for x, y, w, h in bboxes:
            xmin = x / IMAGE_SIZE
            ymin = y / IMAGE_SIZE
            xmax = (x + w) / IMAGE_SIZE
            ymax = (y + h) / IMAGE_SIZE
            converted.append([ymin, xmin, ymax, xmax])
        return converted

    def build_tf_dataset(self, image_dir, ann_path):
        id_to_file, image_to_anns = self.parse_coco_json(ann_path)
        image_ids = list(image_to_anns.keys())

        def gen():
            for img_id in image_ids:
                img_path = os.path.join(image_dir, id_to_file[img_id])
                anns = image_to_anns[img_id]
                bboxes = [ann[0] for ann in anns]
                class_ids = [ann[1] for ann in anns]

                image = self.load_and_preprocess_image(img_path)
                boxes = self.convert_to_yxyx(bboxes)

                yield image, {"boxes": np.array(boxes, dtype=np.float32),
                              "class_ids": np.array(class_ids, dtype=np.int32)}

        output_types = (tf.float32, {"boxes": tf.float32, "class_ids": tf.int32})
        output_shapes = (
            (IMAGE_SIZE, IMAGE_SIZE, 3),
            {"boxes": (None, 4), "class_ids": (None,)}
        )

        ds = tf.data.Dataset.from_generator(gen, output_types=output_types, output_shapes=output_shapes)
        ds = ds.shuffle(512).padded_batch(
            self.batch_size,
            padded_shapes=(
                (IMAGE_SIZE, IMAGE_SIZE, 3),
                {"boxes": [None, 4], "class_ids": [None]}
            ),
            padding_values=(
                0.0,
                {"boxes": tf.constant(0.0, tf.float32), "class_ids": tf.constant(0, tf.int32)}
            )
        ).prefetch(tf.data.AUTOTUNE)

        return ds

    def get_data(self):
        train_img_dir = os.path.join(self.coco_dir, "train2017")
        train_ann_path = os.path.join(self.coco_dir, "annotations", "instances_train2017.json")
        val_img_dir = os.path.join(self.coco_dir, "val2017")
        val_ann_path = os.path.join(self.coco_dir, "annotations", "instances_val2017.json")

        train_ds = self.build_tf_dataset(train_img_dir, train_ann_path)
        val_ds = self.build_tf_dataset(val_img_dir, val_ann_path)
        return train_ds, val_ds


if __name__ == "__main__":
    loader = COCORCNNLoader(batch_size=4)
    train_ds, val_ds = loader.get_data()

    for images, targets in train_ds.take(1):
        print("Image batch:", images.shape)
        print("Boxes batch:", targets['boxes'].shape)
        print("Class IDs batch:", targets['class_ids'].shape)
