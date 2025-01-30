import tensorflow as tf
import numpy as np

from configs.config import T
from diffusion.forward_diffusion import forward_diffusion

def loss_fn(model, x_0, t):
    noise = tf.random.normal(shape=tf.shape(x_0))
    x_t, true_noise = forward_diffusion(x_0, t, noise)

    pred_noise = model(x_t, training=True)
    return tf.reduce_mean(tf.losses.mse(true_noise, pred_noise))

# Training loop
def train(model, dataset, epochs=5, lr=1e-3):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    print("training started...")
    for epoch in range(epochs):
        total_loss = 0
        for x_0 in dataset:
            t = np.random.randint(0, T, size=(x_0.shape[0],))
            with tf.GradientTape() as tape:
                loss = loss_fn(model, x_0, t)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss.numpy()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")


