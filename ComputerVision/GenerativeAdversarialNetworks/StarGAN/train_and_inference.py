import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import cv2
from silence_tensorflow import silence_tensorflow
import time
import gc

from params import *
from model import *
from dataset import *


silence_tensorflow()


parser = argparse.ArgumentParser(description='Select GPU[0-3]:')
parser.add_argument('--gpu', type=int, default=-1,
                    help='GPU number')
args = parser.parse_args()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if args.gpu >= 0 and args.gpu <= 3:
    tf.config.experimental.set_visible_devices(
        physical_devices[args.gpu], 'GPU')

else:
    tf.config.experimental.set_visible_devices(physical_devices[2:], 'GPU')
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=25),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='star_gan.h5', save_weights_only=True, monitor='loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.LearningRateScheduler(
        schedule=lambda epoch: 0.005 * (0.995 ** (epoch//10)))
]

bce = tf.keras.losses.BinaryCrossentropy(
    from_logits=True)
l1 = tf.keras.losses.MeanSquaredError()
gen_optimizer = tf.keras.optimizers.Adam(0.0001, 0.5)
disc_optimizer = tf.keras.optimizers.Adam(0.0001, 0.5)


def gradient_penalty_loss(averaged_output, x_hat):
    gradients = tf.gradients(averaged_output, x_hat)[0]
    gradients_sqr = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(
        gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))  # type: ignore
    gradients_l2_norm = tf.sqrt(gradients_sqr_sum)
    gradient_penalty = tf.square(gradients_l2_norm - 1)
    return tf.reduce_mean(gradient_penalty)


def discriminator_loss(real_output, fake_output, averaged_output, interpolated_img, lamb_gp=10):
    real_loss = -tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    gp_loss = gradient_penalty_loss(averaged_output, interpolated_img)
    total_loss = real_loss + fake_loss + gp_loss * lamb_gp
    return total_loss


def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)


def reconstrution_loss(loss_object, real_image, recon_image, lamb_rec=10):
    return loss_object(real_image, recon_image) * lamb_rec


def domain_classification_loss(loss_object, category, output, lamb_cls=1):
    return loss_object(category, output) * lamb_cls


def random_weighted_average(inputs, alpha):
    return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


@tf.function
def train_generator(generator, discriminator, images, ori_labels, tar_labels):
    with tf.GradientTape(persistent=True) as tape:
        fake_images = generator([images, tar_labels])
        recon_images = generator([fake_images, ori_labels])
        fake_output, fake_class = discriminator(
            fake_images)  # type: ignore
        gen_loss = generator_loss(fake_output)
        fake_class_loss = domain_classification_loss(
            bce, tar_labels, fake_class)
        recon_loss = reconstrution_loss(l1, images, recon_images)
        total_gen_loss = gen_loss + fake_class_loss + recon_loss

    grad_gen = tape.gradient(
        total_gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(
        zip(grad_gen, generator.trainable_variables))

    return fake_images, tf.reduce_mean(gen_loss), tf.reduce_mean(fake_class_loss), tf.reduce_mean(recon_loss)


@tf.function
def train_discriminator(generator, discriminator,  images, ori_labels, tar_labels):
    with tf.GradientTape(persistent=True) as tape:
        real_output, real_class = discriminator(
            images)  # type: ignore
        fake_images = generator([images, tar_labels])
        fake_output, fake_class = discriminator(
            fake_images)  # type: ignore
        alpha = tf.random.uniform(
            (images.shape[0], 1, 1, 1), dtype=tf.float32)
        interpolated_img = random_weighted_average(
            [images, fake_images], alpha)
        averaged_output, _ = discriminator(
            interpolated_img)  # type: ignore

        disc_loss = discriminator_loss(
            real_output, fake_output, averaged_output, interpolated_img)
        real_class_loss = domain_classification_loss(
            bce, ori_labels, real_class)
        total_disc_loss = disc_loss + real_class_loss

    grad_disc = tape.gradient(
        total_disc_loss, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(
        zip(grad_disc, discriminator.trainable_variables))

    return tf.reduce_mean(real_class_loss), tf.reduce_mean(disc_loss)


def save_imgs(epoch, generator, real_x):
    tags = np.array([[0, 1, 0, 1, 0]
                    for i in range(len(real_x))], dtype=np.uint8)
    gene_imgs = generator([real_x, tags], training=False)

    gene_imgs = ((gene_imgs.numpy() + 1) * 127.5).astype(np.uint8)
    real_x = ((real_x.numpy() + 1) * 127.5).astype(np.uint8)

    fig = plt.figure(figsize=(4, 8))

    tmp = 0
    for i in range(0, 4):
        plt.subplot(4, 2, i + 1 + tmp)
        plt.imshow(real_x[i])
        plt.axis('off')
        plt.subplot(4, 2, i + 2 + tmp)
        plt.imshow(gene_imgs[i])
        plt.axis('off')
        tmp += 1

    fig.savefig("trial_images/result_{}.png".format(str(epoch).zfill(5)))


# class callback(tf.keras.callbacks.Callback):
#     def __init__(self, model):
#         super(callback, self).__init__()
#         self.model = model

#     def on_epoch_end(self, epoch, logs=None):
#         self.model.counter += 1
#         print(self.model.counter)


# class StarGan(tf.keras.models.Model):
#     def __init__(self):
#         super(StarGan, self).__init__()
#         self.generator = Generator().show_model()
#         self.generator.summary(expand_nested=True)
#         tf.keras.utils.plot_model(
#             self.generator, to_file=Generator.__name__+'.png', show_shapes=True, expand_nested=True)
#         self.discriminator = Discriminator(IMAGE_SHAPE, NUM_CLASS).show_model()
#         self.discriminator.summary(expand_nested=True)
#         tf.keras.utils.plot_model(
#             self.discriminator, to_file=Discriminator.__name__+'.png', show_shapes=True, expand_nested=True)
#         self.counter = 0

#     @tf.function
#     def train_generator(self, images, ori_labels, tar_labels):
#         with tf.GradientTape(persistent=True) as tape:
#             fake_images = self.generator([images, tar_labels])
#             recon_images = self.generator([fake_images, ori_labels])
#             fake_output, fake_class = self.discriminator(
#                 fake_images)  # type: ignore
#             gen_loss = generator_loss(fake_output)
#             fake_class_loss = domain_classification_loss(
#                 bce, tar_labels, fake_class)
#             recon_loss = reconstrution_loss(l1, images, recon_images)
#             total_gen_loss = gen_loss + fake_class_loss + recon_loss

#         grad_gen = tape.gradient(
#             total_gen_loss, self.generator.trainable_variables)
#         gen_optimizer.apply_gradients(
#             zip(grad_gen, self.generator.trainable_variables))

#         return tf.reduce_mean(gen_loss), tf.reduce_mean(fake_class_loss), tf.reduce_mean(recon_loss)

#     @tf.function
#     def train_discriminator(self, images, ori_labels, tar_labels):
#         with tf.GradientTape(persistent=True) as tape:
#             real_output, real_class = self.discriminator(
#                 images)  # type: ignore
#             fake_images = self.generator([images, tar_labels])
#             fake_output, fake_class = self.discriminator(
#                 fake_images)  # type: ignore
#             alpha = tf.random.uniform(
#                 (images.shape[0], 1, 1, 1), dtype=tf.float32)
#             interpolated_img = random_weighted_average(
#                 [images, fake_images], alpha)
#             averaged_output, _ = self.discriminator(
#                 interpolated_img)  # type: ignore

#             disc_loss = discriminator_loss(
#                 real_output, fake_output, averaged_output, interpolated_img)
#             real_class_loss = domain_classification_loss(
#                 bce, ori_labels, real_class)
#             total_disc_loss = disc_loss + real_class_loss

#         grad_disc = tape.gradient(
#             total_disc_loss, self.discriminator.trainable_variables)
#         disc_optimizer.apply_gradients(
#             zip(grad_disc, self.discriminator.trainable_variables))

#         return tf.reduce_mean(real_class_loss), tf.reduce_mean(disc_loss)

#     def train_step(self, inputs):
#         images, ori_labels, tar_labels = inputs
#         real_cls_loss, disc_loss = self.train_discriminator(
#             images, ori_labels, tar_labels)  # type: ignore
#         gen_loss = 0
#         recon_loss = 0
#         if self.counter % N_CRITIC == 0:
#             gen_loss, fake_cls_loss, recon_loss = self.train_generator(
#                 images, ori_labels, tar_labels)  # type: ignore

#         return {'G Loss': gen_loss, 'D loss': disc_loss, 'R loss': recon_loss}


def train_model(train, epochs=ITERATION, batch_size=BATCH_SIZE):
    generator = Generator().show_model()
    generator.summary()
    tf.keras.utils.plot_model(
        generator, to_file=Generator.__name__+'.png', show_shapes=True, expand_nested=True)
    discriminator = Discriminator(IMAGE_SHAPE, NUM_CLASS).show_model()
    discriminator.summary()
    tf.keras.utils.plot_model(
        discriminator, to_file=Discriminator.__name__+'.png', show_shapes=True, expand_nested=True)

    try:
        generator.load_weights('trial_weights/generator.h5')
        discriminator.load_weights('trial_weights/discriminator.h5')
    except FileNotFoundError:
        os.makedirs('trial_weights', exist_ok=True)
        os.makedirs('trial_images', exist_ok=True)

    print('Training...')
    for epoch in range(epochs+1):
        start = time.time()
        gc.collect()
        total_real_cls_loss = 0
        total_disc_loss = 0

        total_gen_loss = 0
        total_fake_cls_loss = 0
        total_recon_loss = 0

        for images, ori_labels, tar_labels in tqdm.tqdm(train):
            real_cls_loss, disc_loss = train_discriminator(
                generator, discriminator, images, ori_labels, tar_labels)  # type: ignore
            total_real_cls_loss += real_cls_loss
            total_disc_loss += disc_loss

            if epoch % N_CRITIC == 0:
                fake_images, gen_loss, fake_cls_loss, recon_loss = train_generator(
                    generator, discriminator, images, ori_labels, tar_labels)  # type: ignore
                total_gen_loss += gen_loss
                total_fake_cls_loss += fake_cls_loss
                total_recon_loss += recon_loss

        log = 'Time for epoch {}/{} is {} sec : - disc_loss = {}, real_cls_loss = {}'.format(
            epoch + 1, epochs, time.time() - start, total_disc_loss/batch_size, total_real_cls_loss/batch_size)
        log += 'gen_loss = {}, fake_cls_loss = {}, recon_loss = {}'.format(
            total_gen_loss/batch_size, total_fake_cls_loss/batch_size, total_recon_loss/batch_size)

        if epoch % EVERY_STEP == 0:
            final_image = []
            temp_true = []
            temp_fake = []

            # type: ignore
            true_im = np.array(images[:18].numpy()*255.0, dtype=np.uint8)
            fake_im = np.array(fake_images[:18].numpy()*255.0, dtype=np.uint8)
            for idx, (orig_img, fake_img) in enumerate(zip(true_im, fake_im)):
                temp_true.append(orig_img)
                temp_fake.append(fake_img)
                if len(temp_fake) == 6:
                    final_image.append(temp_true)
                    final_image.append(temp_fake)
                    temp_true = []
                    temp_fake = []

            for ind in range(len(final_image)):
                final_image[ind] = np.hstack(final_image[ind])

            img = np.vstack(final_image)
            plt.imsave('trial_images/'+str(epoch)+'.png', img)
            generator.save_weights('trial_weights/generator.h5')
            discriminator.save_weights('trial_weights/discriminator.h5')
        print(log)


if __name__ == '__main__':
    data = dataset()
    train_model(data)
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope() as scope:
    # model = StarGan()
    # model.compile(loss=None, run_eagerly=True)
    # model.fit(data, epochs=10, callbacks=[callback(model)])
