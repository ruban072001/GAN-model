import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv2D, LeakyReLU, UpSampling2D, Reshape, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
import os
import matplotlib.pyplot as plt

# Load the Fashion MNIST dataset
ds = tfds.load('fashion_mnist', split='train', as_supervised=True)

# Preprocess the images: scale to [-1, 1] (for tanh activation)
def scale_img(image, label):
    image = tf.cast(image, tf.float32) / 127.5 - 1.0  # Scale image to [-1, 1]
    return image

# Prepare the dataset
train_f_modi = ds.map(scale_img).cache().shuffle(60000).batch(128).prefetch(tf.data.AUTOTUNE)

# Generator model
def img_generator(): 
    model = Sequential()

    model.add(Dense(7 * 7 * 128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7, 7, 128)))

    model.add(UpSampling2D())
    model.add(Conv2D(128, 4, padding='same'))
    model.add(BatchNormalization())  # Added Batch Normalization
    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D())
    model.add(Conv2D(128, 4, padding='same'))
    model.add(BatchNormalization())  # Added Batch Normalization
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, 4, padding='same'))
    model.add(BatchNormalization())  # Added Batch Normalization
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, 4, padding='same'))
    model.add(BatchNormalization())  # Added Batch Normalization
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(64, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(32, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    # Change to 'tanh' activation for the output layer (for [-1, 1] output range)
    model.add(Conv2D(1, 4, padding='same', activation='tanh'))  
    return model


# Discriminator model
def img_discriminator():
    model = Sequential()
    model.add(Conv2D(32, 4, input_shape=(28, 28, 1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, 4))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, 4))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, 4))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    return model

generator = img_generator()
discriminator = img_discriminator()

# Optimizers and Losses
g_opt = Adam(learning_rate=0.0001)
d_opt = Adam(learning_rate=0.00001)
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()

# GAN class with gradient clipping and label smoothing
class Gan:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

    def compil_gan(self, g_opt, d_opt, g_loss, d_loss):
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_gan(self, batch):
        real_img = batch
        fake_img = self.generator(tf.random.normal((tf.shape(real_img)[0], 128)), training=False)

        # Add noise to real and fake images for stability
        real_img += tf.random.normal(shape=real_img.shape, mean=0.0, stddev=0.1)
        fake_img += tf.random.normal(shape=fake_img.shape, mean=0.0, stddev=0.1)

        # Discriminator training
        with tf.GradientTape() as d_tape:
            yhat_real = self.discriminator(real_img, training=True)
            yhat_fake = self.discriminator(fake_img, training=True)

            # Label smoothing: real labels are slightly less than 1
            y_realfake = tf.concat([tf.ones_like(yhat_real) * 0.9, tf.zeros_like(yhat_fake)], axis=0)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            total_d_loss = self.d_loss(y_realfake, yhat_realfake)

        # Update discriminator with gradient clipping
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        dgrad = [tf.clip_by_value(grad, -1.0, 1.0) for grad in dgrad]  # Clip gradients
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        # Generator training
        with tf.GradientTape() as g_tape:
            gen_img = self.generator(tf.random.normal((tf.shape(real_img)[0], 128)), training=True)
            predicted_labels = self.discriminator(gen_img, training=False)
            total_g_loss = self.g_loss(tf.ones_like(predicted_labels) * 0.9, predicted_labels)  # Use smooth labels

        # Update generator with gradient clipping
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        ggrad = [tf.clip_by_value(grad, -1.0, 1.0) for grad in ggrad]  # Clip gradients
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {'d_loss': total_d_loss, 'g_loss': total_g_loss}

# Instantiate GAN
f_g = Gan(generator, discriminator)
f_g.compil_gan(g_opt, d_opt, g_loss, d_loss)

# Function to save generated images
def save_generated_images(generator, epoch, num_images=10, save_dir='generated_images'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    noise = tf.random.normal([num_images, 128])  # Generate random noise
    generated_images = generator(noise, training=False)
    
    # Rescale images back to [0, 1] for saving
    generated_images = (generated_images + 1) / 2.0

    fig, axs = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    for i in range(num_images):
        axs[i].imshow(generated_images[i, :, :, 0], cmap='gray')
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/generated_epoch_{epoch + 1}.png")
    plt.close()

# Training the GAN
epochs = 100
for epoch in range(epochs):
    d_loss_total = 0
    g_loss_total = 0
    batch_count = 0

    for batch in train_f_modi:
        losses = f_g.train_gan(batch)
        d_loss_total += losses['d_loss']
        g_loss_total += losses['g_loss']
        batch_count += 1

    d_loss_avg = d_loss_total / batch_count
    g_loss_avg = g_loss_total / batch_count

    print(f"Epoch {epoch + 1}/{epochs}, Discriminator Loss: {d_loss_avg:.4f}, Generator Loss: {g_loss_avg:.4f}")
    print("=" * 50)

    save_generated_images(generator, epoch)
