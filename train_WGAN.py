import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
file_path = "input_transaction_data.csv"
df = pd.read_csv(file_path).select_dtypes(include=[np.number])

# Standardize data for improved distribution matching
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

X_train, _ = train_test_split(df_scaled, test_size=0.2, random_state=42)
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)

# GAN parameters
latent_dim = 128  # Increased latent space
tau = 0.01  # Temperature parameter for stabilization
num_features = X_train.shape[1]

def make_generator():
    model = tf.keras.Sequential([
        layers.Dense(2048, input_shape=(latent_dim,), kernel_initializer='he_normal'),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(1024, kernel_initializer='he_normal'),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),

        layers.Dense(512, kernel_initializer='he_normal'),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),

        layers.Dense(num_features, activation='linear')  # Output remains in standardized scale
    ])
    return model

def make_discriminator():
    model = tf.keras.Sequential([
        layers.Dense(512, input_shape=(num_features,), kernel_initializer='he_normal'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),

        layers.Dense(256, kernel_initializer='he_normal'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),

        layers.Dense(128, kernel_initializer='he_normal'),
        layers.LeakyReLU(alpha=0.2),

        layers.Dense(1)
    ])
    return model

class WGAN(tf.keras.Model):
    def __init__(self, generator, discriminator, gp_weight=30):
        super(WGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.gp_weight = gp_weight

    def compile(self, g_optimizer, d_optimizer):
        super(WGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def gradient_penalty(self, real_data, fake_data):
        alpha = tf.random.uniform([tf.shape(real_data)[0], 1], 0., 1.)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        grads = tape.gradient(pred, [interpolated])[0]
        penalty = tf.reduce_mean((tf.norm(grads, axis=1) - 1.0) ** 2)
        return self.gp_weight * penalty

    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        noise = tf.random.normal([batch_size, latent_dim])

        for _ in range(5):  # Train Discriminator multiple times per Generator update
            fake_data = self.generator(noise, training=True)
            with tf.GradientTape() as tape:
                real_preds = self.discriminator(real_data, training=True)
                fake_preds = self.discriminator(fake_data, training=True)
                gp = self.gradient_penalty(real_data, fake_data)
                d_loss = tf.reduce_mean(fake_preds) - tf.reduce_mean(real_preds) + gp
            d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            fake_data = self.generator(noise, training=True)
            fake_preds = self.discriminator(fake_data, training=True)
            g_loss = -tf.reduce_mean(fake_preds)
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {'d_loss': d_loss, 'g_loss': g_loss}

generator = make_generator()
discriminator = make_discriminator()

g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

wgan = WGAN(generator, discriminator)
wgan.compile(g_optimizer=g_optimizer, d_optimizer=d_optimizer)

num_epochs = 50
batch_size = 64
save_interval = 5

for epoch in range(1, num_epochs + 1):
    wgan.fit(X_train, epochs=1, batch_size=batch_size)

    if epoch % save_interval == 0:
        generator.save(f"/content/drive/MyDrive/Training_Dataset/wgan_generator_epoch_{epoch}.h5")
        print(f"💾 Model Checkpoint Saved at Epoch {epoch}")

generator.save("/content/drive/MyDrive/Training_Dataset/wgan_generator_final.h5")
print("✅ Final Generator Model Saved!")

noise = tf.random.normal([len(df_scaled) * 10, latent_dim])
synthetic_data = generator.predict(noise)
synthetic_data = scaler.inverse_transform(synthetic_data)

df_synthetic = pd.DataFrame(synthetic_data, columns=df.columns)
synthetic_file = "/content/drive/MyDrive/Training_Dataset/synthetic_data_10x_fixed.csv"
df_synthetic.to_csv(synthetic_file, index=False)
print(f"✅ Synthetic Dataset Generated & Saved: {synthetic_file}")
