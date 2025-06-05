import silence_tensorflow.auto
import tensorflow as tf
import json
import os
import numpy as np

from config import *

from collections import Counter

def compute_class_weights(annotation_path, num_classes=NUM_CLASSES):
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    label_counts = Counter()
    for ann in data['annotations']:
        label_counts[ann['category_id']] += 1

    counts = np.array([label_counts.get(i, 1) for i in range(num_classes)], dtype=np.float32)
    weights = np.log(1.0 + counts.sum() / counts)
    weights = weights / weights.mean()
    return tf.constant(weights, dtype=tf.float32)



class COCODataset:
    def __init__(self, coco_dir="../coco", batch_size=BATCH_SIZE):
        self.coco_dir = coco_dir
        self.batch_size = batch_size

    def parse_coco_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        id_to_filename = {img["id"]: img["file_name"]
                          for img in data["images"]}
        image_to_anns = {}
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            bbox = ann["bbox"]  # [x, y, w, h]
            category_id = ann["category_id"]
            if image_id not in image_to_anns:
                image_to_anns[image_id] = []
            image_to_anns[image_id].append((bbox, category_id))
        return id_to_filename, image_to_anns

    def load_and_preprocess_image(self, img_path):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        # image = tf.cast(image, tf.float32) / 255.0
        return image

    def encode_label(self, bboxes, labels):
        label_tensor = np.zeros((S, S, B * 5 + NUM_CLASSES), dtype=np.float32)
        for bbox, label in zip(bboxes, labels):
            x, y, w, h = bbox
            cx = x + w / 2
            cy = y + h / 2
            cx_rel = cx / IMAGE_SIZE
            cy_rel = cy / IMAGE_SIZE
            w_rel = w / IMAGE_SIZE
            h_rel = h / IMAGE_SIZE
            grid_x = int(cx_rel * S)
            grid_y = int(cy_rel * S)
            if grid_x >= S or grid_y >= S:
                continue
            if label_tensor[grid_y, grid_x, 4] == 0:
                label_tensor[grid_y, grid_x, 0:5] = [cx_rel * S -
                                                     grid_x, cy_rel * S - grid_y, w_rel, h_rel, 1.0]
                label_tensor[grid_y, grid_x, B * 5 + label] = 1.0
        return label_tensor

    def build_tf_dataset(self, image_dir, ann_path):
        id_to_file, anns = self.parse_coco_json(ann_path)
        image_ids = list(anns.keys())

        def gen():
            for img_id in image_ids:
                img_path = os.path.join(image_dir, id_to_file[img_id])
                bboxes = [ann[0] for ann in anns[img_id]]
                labels = [ann[1] for ann in anns[img_id]]
                image = self.load_and_preprocess_image(img_path)
                label = self.encode_label(bboxes, labels)
                yield image, label

        ds = tf.data.Dataset.from_generator(
            gen,
            output_types=(tf.float32, tf.float32),
            output_shapes=((IMAGE_SIZE, IMAGE_SIZE, 3),
                           (S, S, B * 5 + NUM_CLASSES))
        )
        return ds.shuffle(128).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def get_data(self):
        train_img_dir = os.path.join(self.coco_dir, "train2017")
        train_ann_path = os.path.join(
            self.coco_dir, "annotations", "instances_train2017.json")
        val_img_dir = os.path.join(self.coco_dir, "val2017")
        val_ann_path = os.path.join(
            self.coco_dir, "annotations", "instances_val2017.json")
        train_ds = self.build_tf_dataset(train_img_dir, train_ann_path)
        val_ds = self.build_tf_dataset(val_img_dir, val_ann_path)
        return train_ds, val_ds
    

class COCOClassificationDataset:
    def __init__(self, coco_dir="../coco", batch_size=BATCH_SIZE):
        self.coco_dir = coco_dir
        self.batch_size = batch_size

    def parse_coco_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
        image_to_class = {}

        for ann in data["annotations"]:
            img_id = ann["image_id"]
            class_id = ann["category_id"]
            if img_id not in image_to_class:
                image_to_class[img_id] = []
            image_to_class[img_id].append(class_id)

        # Assign the most frequent class per image
        final_labels = {
            img_id: Counter(class_list).most_common(1)[0][0]
            for img_id, class_list in image_to_class.items()
        }

        return id_to_filename, final_labels

    def load_and_preprocess_image(self, img_path):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        # image = tf.cast(image, tf.float32) / 255.0
        return image

    def build_tf_dataset(self, image_dir, ann_path):
        id_to_file, image_to_label = self.parse_coco_json(ann_path)
        image_ids = list(image_to_label.keys())

        def gen():
            for img_id in image_ids:
                img_path = os.path.join(image_dir, id_to_file[img_id])
                image = self.load_and_preprocess_image(img_path)
                label = image_to_label[img_id]
                yield image, label

        ds = tf.data.Dataset.from_generator(
            gen,
            output_types=(tf.float32, tf.int32),
            output_shapes=((IMAGE_SIZE, IMAGE_SIZE, 3), ())
        )

        ds = ds.shuffle(512).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
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
    dataset_builder = COCODataset(batch_size=4)
    train_ds, val_ds = dataset_builder.get_data()
    # weights = compute_class_weights("./coco/annotations/instances_train2017.json")
    # print(weights)
    
    for images, labels in train_ds.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        # print("Sample label grid:", labels[0])
    
    for images, labels in val_ds.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        # print("Sample label grid:", labels[0])
        
    dataset_builder = COCOClassificationDataset(batch_size=4)
    train_ds, val_ds = dataset_builder.get_data()
    # weights = compute_class_weights("./coco/annotations/instances_train2017.json")
    # print(weights)
    
    for images, labels in train_ds.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        # print("Sample label grid:", labels[0])
    
    for images, labels in val_ds.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        # print("Sample label grid:", labels[0])
