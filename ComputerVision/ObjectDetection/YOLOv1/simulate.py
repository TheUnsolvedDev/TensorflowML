import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from dataset import COCODataset
from model import build_yolo_v1
from config import *


def draw_boxes(image, prediction, score_threshold=0.1):
    h, w, _ = image.shape
    image = image.copy()
    cell_size = IMAGE_SIZE // S

    for i in range(S):
        for j in range(S):
            cell = prediction[i, j]
            conf = cell[4]
            if conf < score_threshold:
                continue

            bx = (cell[0] + j) * cell_size
            by = (cell[1] + i) * cell_size
            bw = cell[2] * IMAGE_SIZE
            bh = cell[3] * IMAGE_SIZE

            x1 = int(bx - bw / 2)
            y1 = int(by - bh / 2)
            x2 = int(bx + bw / 2)
            y2 = int(by + bh / 2)

            class_id = np.argmax(cell[5:])
            label = str(class_id)
            color = (0, 255, 0)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    return image


def simulate():
    # Load dataset
    dataset = COCODataset()
    train_ds, _ = dataset.get_data()

    # Load model
    model = build_yolo_v1()
    model.load_weights(CHECKPOINT_PATH)

    # Take one sample
    for images, _ in train_ds.take(6):
        preds = model.predict(images)
        print("Pred shape:", preds.shape)
        sample = preds[0]
        center_cell = sample[S // 2, S // 2]
        print("Sample center cell prediction:", center_cell)
        print("Max confidence score in prediction:", tf.reduce_max(sample[..., 4:5]).numpy())
        img = images[0].numpy()
        pred = preds[0]

        img = (img * 255).astype(np.uint8)
        annotated = draw_boxes(img, pred)

        plt.figure(figsize=(8, 8))
        plt.imshow(annotated)
        plt.axis('off')
        plt.title("YOLOv1 Prediction")
        plt.show()
        # break 


if __name__ == "__main__":
    simulate()
