import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from config import *
from model import *
from dataset import *


def train(
    start_res=START_RES,
    target_res=TARGET_RES,
    steps_per_epoch=5000,
    display_images=True,
):
    opt_cfg = {"learning_rate": 1e-3, "beta_1": 0.0,
               "beta_2": 0.99, "epsilon": 1e-8}

    dataset = Dataset()
    style_gan = StyleGAN(start_res=START_RES, target_res=TARGET_RES)

    val_batch_size = 16
    val_z = tf.random.normal((val_batch_size, style_gan.z_dim))
    val_noise = style_gan.generate_noise(val_batch_size)

    start_res_log2 = int(np.log2(start_res))
    target_res_log2 = int(np.log2(target_res))

    for res_log2 in range(start_res_log2, target_res_log2 + 1):
        res = 2 ** res_log2
        for phase in ["TRANSITION", "STABLE"]:
            if res == start_res and phase == "TRANSITION":
                continue

            train_dl = dataset.create_dataloader(res)

            steps = int(dataset.train_step_ratio[res_log2] * steps_per_epoch)

            style_gan.compile(
                d_optimizer=tf.keras.optimizers.legacy.Adam(**opt_cfg),
                g_optimizer=tf.keras.optimizers.legacy.Adam(**opt_cfg),
                loss_weights={"gradient_penalty": 10, "drift": 0.001},
                steps_per_epoch=steps,
                res=res,
                phase=phase,
                run_eagerly=False,
            )

            prefix = f"res_{res}x{res}_{style_gan.phase}"

            ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
                f"checkpoints/stylegan_{res}x{res}.ckpt",
                save_weights_only=True,
                verbose=0,
            )
            print(phase)
            style_gan.fit(
                train_dl, epochs=1, steps_per_epoch=steps, callbacks=[ckpt_cb]
            )

            if display_images:
                images = style_gan(
                    {"z": val_z, "noise": val_noise, "alpha": 1.0})
                plot_images(images, res_log2, fname=f"{prefix}.png")


if __name__ == "__main__":
    # train(start_res=4, target_res=64, steps_per_epoch=1000, display_images=True)

    style_gan = StyleGAN(start_res=START_RES, target_res=TARGET_RES)
    style_gan.grow_model(128)
    style_gan.load_weights(os.path.join("pretrained/stylegan_128x128.ckpt"))

    tf.random.set_seed(196)
    batch_size = 2
    z = tf.random.normal((batch_size, style_gan.z_dim))
    w = style_gan.mapping(z)
    noise = style_gan.generate_noise(batch_size=batch_size)
    images = style_gan({"style_code": w, "noise": noise, "alpha": 1.0})
    plot_images(images, 5)
    
    alpha = 0.4
    w_mix = np.expand_dims(alpha * w[0] + (1 - alpha) * w[1], 0)
    noise_a = [np.expand_dims(n[0], 0) for n in noise]
    mix_images = style_gan({"style_code": w_mix, "noise": noise_a})
    image_row = np.hstack([images[0], images[1], mix_images[0]])
    plt.figure(figsize=(9, 3))
    plt.imshow(image_row)
    plt.axis("off")
    plt.show()
