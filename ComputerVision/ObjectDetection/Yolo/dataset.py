import numpy as np
import cv2
import tqdm
import silence_tensorflow.auto
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import json

from config import *
tf.config.run_functions_eagerly(True)

def plot_images(dataset,type = 'train'):
    LOC = TRAIN_IMAGES if type == 'train' else VAL_IMAGES
    for key in tqdm.tqdm(dataset['images'].keys()):
        image_location = dataset['images'][key]
        image = cv2.imread(f'{LOC}{image_location}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image).astype(np.int32)
        for clas in dataset['annotations'][key].keys():
            for bbox in dataset['annotations'][key][clas]:
                bbox = np.array(bbox).astype(np.int32)
                cv2.rectangle(
                    image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
                cv2.putText(image, COCO_LABELS[COCO_FAULT[clas]], (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        plt.imshow(image)
        plt.show()


def sanity_check_annotations(json_file, dataset):
    from collections import defaultdict
    classes_count = defaultdict(int)
    for key in tqdm.tqdm(dataset['annotations'].keys()):
        for clas in dataset['annotations'][key].keys():
            for bbox in dataset['annotations'][key][clas]:
                classes_count[clas] += 1
    total_objects_dict = sorted(
        classes_count.items(), key=lambda x: x[0], reverse=False)
    total_objects = sum([x[1] for x in total_objects_dict])
    json_length = len(json_file['annotations'])
    # print('Total classes: ',len(classes_count.keys()),classes_count)
    assert total_objects == json_length


class DatasetPrep:
    def __init__(self):
        self.train_json_file = json.load(open(TRAIN_JSON))
        self.val_json_file = json.load(open(VAL_JSON))

        self.train_dataset = {
            'images': {},
            'annotations': {},
        }

        self.val_dataset = {
            'images': {},
            'annotations': {},
        }

    def prep_train(self):
        for i in range(len(self.train_json_file['images'])):
            key_type = 'images'
            key_image_id = self.train_json_file['images'][i]['id']
            self.train_dataset[key_type][key_image_id] = self.train_json_file['images'][i]['file_name']

            key_type = 'annotations'
            key_image_id = self.train_json_file['images'][i]['id']
            self.train_dataset[key_type][key_image_id] = {}

        for i in range(len(self.train_json_file['annotations'])):
            key_type = 'annotations'
            key_image_id = self.train_json_file['annotations'][i]['image_id']
            key_category_id = self.train_json_file['annotations'][i]['category_id']
            try:
                self.train_dataset[key_type][key_image_id][key_category_id].append(
                    self.train_json_file['annotations'][i]['bbox'])
            except KeyError:
                self.train_dataset[key_type][key_image_id][key_category_id] = [
                ]
                self.train_dataset[key_type][key_image_id][key_category_id].append(
                    self.train_json_file['annotations'][i]['bbox'])

        
        with open('coco/train.txt', 'w') as f:
            for key in tqdm.tqdm(self.train_dataset['images'].keys()):
                image_location = self.train_dataset['images'][key]
                annotation_data = self.train_dataset['annotations'][key]
                row = f'{TRAIN_IMAGES}{image_location} '

                for clas in annotation_data.keys():
                    for bbox in annotation_data[clas]:
                        bbox = np.array(bbox).astype(np.int32)
                        box_path = f'{bbox[0]},{bbox[1]},{
                            bbox[2]},{bbox[3]},{clas} '
                        row += box_path
                f.write(row + '\n')

        # plot_images(self.train_dataset,type='train')
        sanity_check_annotations(self.train_json_file, self.train_dataset)

    def prep_val(self):
        for i in range(len(self.val_json_file['images'])):
            key_type = 'images'
            key_image_id = self.val_json_file['images'][i]['id']
            self.val_dataset[key_type][key_image_id] = self.val_json_file['images'][i]['file_name']

            key_type = 'annotations'
            key_image_id = self.val_json_file['images'][i]['id']
            self.val_dataset[key_type][key_image_id] = {}

        for i in range(len(self.val_json_file['annotations'])):
            key_type = 'annotations'
            key_image_id = self.val_json_file['annotations'][i]['image_id']
            key_category_id = self.val_json_file['annotations'][i]['category_id']
            try:
                self.val_dataset[key_type][key_image_id][key_category_id].append(
                    self.val_json_file['annotations'][i]['bbox'])
            except KeyError:
                self.val_dataset[key_type][key_image_id][key_category_id] = []
                self.val_dataset[key_type][key_image_id][key_category_id].append(
                    self.val_json_file['annotations'][i]['bbox'])

        with open('coco/val.txt', 'w') as f:
            for key in tqdm.tqdm(self.val_dataset['images'].keys()):
                image_location = self.val_dataset['images'][key]
                annotation_data = self.val_dataset['annotations'][key]
                row = f'{VAL_IMAGES}{image_location} '

                for clas in annotation_data.keys():
                    for bbox in annotation_data[clas]:
                        bbox = np.array(bbox).astype(np.int32)
                        box_path = f'{bbox[0]},{bbox[1]},{
                            bbox[2]},{bbox[3]},{clas} '
                        row += box_path
                f.write(row + '\n')

        # plot_images(self.val_dataset,type='val')
        sanity_check_annotations(self.val_json_file, self.val_dataset)
        

# class Dataset:
#     def __init__(self, train_img_size = IMAGE_SIZE, val_img_size = IMAGE_SIZE, num_classes = len(COCO_LABELS),yolo_anchors = YOLO_ANCHORS,strides = YOLO_STRIDES,anchor_per_scale = YOLO_ANCHORS_PER_SCALE,max_bbox_per_scale = YOLO_MAX_BBOX_PER_SCALE,batch_size = BATCH_SIZE):
#         self.train_img_size = train_img_size
#         self.val_img_size = val_img_size
#         self.num_classes = num_classes
#         self.strides = np.array(strides)
#         self.anchors = (np.array(yolo_anchors).T/self.strides).T
#         self.anchor_per_scale = anchor_per_scale
#         self.max_bbox_per_scale = max_bbox_per_scale
        
# def calculate_iou(box1, box2):
#     intersect_w = tf.minimum(box1[0], box2[0])
#     intersect_h = tf.minimum(box1[1], box2[1])
#     intersection = intersect_w * intersect_h
#     area1 = box1[0] * box1[1]
#     area2 = box2[0] * box2[1]
#     union = area1 + area2 - intersection
#     return intersection / union

        
# def parse_annotation(annotation):
#     parts = tf.strings.split(annotation)
#     image_path = parts[0]
#     labels = tf.strings.to_number(parts[1:])
#     labels = tf.reshape(labels, [-1, 5])  # Each box: [x, y, w, h, class_id]
#     return image_path, labels

# def preprocess_image(image_path, labels):
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
#     # image = image / 255.0  # Normalize to [0, 1]
#     labels = labels * [IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1]
#     return image, labels



# @tf.function
# def preprocess_labels(labels):
#     # Prepare empty grids for small, medium, and large scales
#     small_grid = tf.zeros((IMAGE_SIZE // YOLO_STRIDES[0], IMAGE_SIZE // YOLO_STRIDES[0], 3, 5 + NUM_CLASSES))
#     medium_grid = tf.zeros((IMAGE_SIZE // YOLO_STRIDES[1], IMAGE_SIZE // YOLO_STRIDES[1], 3, 5 + NUM_CLASSES))
#     large_grid = tf.zeros((IMAGE_SIZE // YOLO_STRIDES[2], IMAGE_SIZE // YOLO_STRIDES[2], 3, 5 + NUM_CLASSES))
#     grids = [small_grid, medium_grid, large_grid]

#     # Extract bounding box data
#     x, y, w, h, class_id = tf.split(labels, [1, 1, 1, 1, 1], axis=-1)

#     for i, (grid, stride, anchors) in enumerate(zip(grids, YOLO_STRIDES, YOLO_ANCHORS)):
#         grid_size = IMAGE_SIZE // stride
#         x_cell = tf.cast(x // stride, tf.int32)
#         y_cell = tf.cast(y // stride, tf.int32)

#         for j, anchor in enumerate(anchors):
#             anchor_w, anchor_h = anchor

#             # Compute IOU between box and anchor
#             box_wh = tf.concat([w, h], axis=-1)
#             anchor_wh = tf.constant([anchor_w, anchor_h], dtype=tf.float32)
#             iou = calculate_iou(box_wh, anchor_wh)

#             # Use tf.cond to decide whether to squeeze or not
#             mask = tf.cond(
#                 tf.equal(tf.shape(iou)[-1], 1),
#                 lambda: tf.squeeze(iou, axis=-1),
#                 lambda: iou
#             )

#             # Create a valid mask based on the threshold
#             mask = mask > YOLO_IOU_LOSS_THRESH

#             # Update grid cells with valid boxes
#             updates = tf.concat(
#                 [x, y, w, h, tf.ones_like(x), tf.one_hot(tf.cast(class_id, tf.int32), NUM_CLASSES)],
#                 axis=-1,
#             )
#             updates = tf.boolean_mask(updates, mask)
#             x_indices = tf.boolean_mask(x_cell, mask)
#             y_indices = tf.boolean_mask(y_cell, mask)

#             grid = tf.tensor_scatter_nd_update(
#                 grid,
#                 tf.stack([y_indices, x_indices, tf.fill_like(y_indices, j)], axis=-1),
#                 updates,
#             )

#         grids[i] = grid

#     return tuple(grids)




        
        
# def prepare_dataset(annotation_file, batch_size):
#     dataset = tf.data.TextLineDataset(annotation_file)
#     dataset = dataset.map(parse_annotation)
#     dataset = dataset.map(lambda img_path, lbl: (preprocess_image(img_path, lbl)))
#     dataset = dataset.map(lambda img, lbl: (img, preprocess_labels(lbl)))
#     dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     return dataset

def parse_annotation(annotation_line):
    parts = tf.strings.split(annotation_line)
    image_path = parts[0]
    
    # Read the image file
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # Parse annotations
    annotations = parts[1:]
    bboxes = []
    class_ids = []
    
    for ann in annotations:
        values = tf.strings.split(ann, sep=',')
        x, y, w, h, class_id = tf.strings.to_number(values, out_type=tf.float32)
        bboxes.append([x, y, w, h])
        class_ids.append(class_id)
    
    return image, (tf.stack(bboxes), tf.stack(class_ids))

# def parse_annot()

def preprocess_yolo(image, bboxes, class_ids, input_size=416):
    image = tf.image.resize(image, (input_size, input_size))
    image_shape = tf.shape(image)[:2]
    bboxes = bboxes * tf.cast(tf.tile(image_shape, [2]), tf.float32)
    return image, (bboxes, class_ids)


def main():
    # obj = DatasetPrep()
    # obj.prep_train()
    # obj.prep_val()

    # data = prepare_dataset('train.txt', BATCH_SIZE)
    
    image_loc, annotations = [],[]
    with open('coco/train.txt', 'r') as f:
        for line in f.readlines():
            image_loc.append(line.split()[0])
            bboxes = line.split()[1:]
            bboxes = np.array([list(map(int, box.split(','))) for box in bboxes])
            annotations.append(bboxes)
    print(image_loc[0], annotations[0])
    data = tf.data.Dataset.from_tensor_slices((image_loc, annotations))
    

if __name__ == '__main__':
    main()
