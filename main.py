import os
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from tqdm import tqdm

T = 250
beta = np.linspace(1e-4, 0.02, T)
alpha = 1 - beta
alpha_bar = np.cumprod(alpha)

### Functies voor het inladen of creëren van de data

def preprocess_and_save_images(image_dir, save_dir, img_size):
    """
    Laadt afbeeldingen, schaalt ze naar 128x128 en slaat ze op in een nieuwe map.

    :param image_dir: Map met originele afbeeldingen.
    :param save_dir: Map om de voorbewerkte afbeeldingen op te slaan.
    :param img_size: Gewenste grootte van de afbeeldingen.
    :return: NumPy-array met voorbewerkte afbeeldingen.
    """
    images = []
    print(f"Start met laden en preprocessen van afbeeldingen uit {image_dir}...")

    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path).convert("RGB")
            img = img.resize(img_size)  # Rescale naar 128x128
            img_array = np.array(img) / 127.5 - 1  # Normaliseren naar [-1, 1]
            images.append(img_array)

            # Opslaan van de bewerkte afbeelding in de nieuwe map
            save_path = os.path.join(save_dir, f"baroque_{i:04d}.png")
            img_pil = Image.fromarray(((img_array + 1) * 127.5).astype(np.uint8))  # Terug naar [0,255] voor opslag
            img_pil.save(save_path)

    images_np = np.array(images)
    print(f"{len(images_np)} afbeeldingen verwerkt en opgeslagen in {save_dir}!")
    
    return images_np


def load_images_from_folder(image_dir):
    """
    Laadt alle afbeeldingen uit een map en converteert ze naar een genormaliseerde NumPy-array.
    
    :param image_dir: Pad naar de map met opgeslagen afbeeldingen.
    :return: NumPy-array met afbeeldingen (genormaliseerd naar [-1, 1])
    """
    images = []
    for filename in sorted(os.listdir(image_dir)):  # Zorgt voor juiste volgorde
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img) / 127.5 - 1  # Normaliseren naar [-1, 1]
            images.append(img_array)

    images_np = np.array(images)
    print(f"{len(images_np)} afbeeldingen geladen uit {image_dir}!")
    return images_np


def show_images(dataset, num_images=9):
    """
    Toont een aantal willekeurige afbeeldingen uit de dataset in een raster.
    
    :param dataset: np.array van afbeeldingen met vorm (aantal, 256, 256, 3)
    :param num_images: Aantal afbeeldingen om te tonen (moet een kwadraatgetal zijn, bijv. 4, 9, 16)
    """
    num_cols = int(np.sqrt(num_images))  # Raster met even aantal kolommen en rijen
    num_rows = int(np.ceil(num_images / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    axes = axes.flatten()  # Maakt een lijst van subplot assen
    
    indices = np.random.choice(len(dataset), num_images, replace=False)  # Willekeurige indices
    
    for i, idx in enumerate(indices):
        img = (dataset[idx] + 1) / 2  # Terugschalen naar [0, 1] voor correcte weergave
        axes[i].imshow(img)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


### Functies voor het model.

def forward_diffusion(x0, t, noise):
    sqrt_alpha_bar_t = np.float32(math.sqrt(alpha_bar[t]))
    sqrt_one_minus_alpha_bar_t = np.float32(math.sqrt(1 - alpha_bar[t]))
    return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise


def get_t_embedding(t, dim):
    # Normalize the timestep and convert to an array.
    t = np.array([t], dtype=np.float32)
    t = t / T
    pe = np.zeros(dim, dtype=np.float32)
    
    # Use t[0] because math.sin and math.cos require scalar inputs.
    for i in range(0, dim, 2):
        pe[i] = math.sin(t[0] * (10000 ** (-i / dim)))
        if i + 1 < dim:
            pe[i + 1] = math.cos(t[0] * (10000 ** (-i / dim)))
    
    return pe


def layer_norm(x, eps=1e-5):
    """
    Apply layer normalization over the last dimension.
    """
    mean, variance = tf.nn.moments(x, axes=[1], keepdims=True)
    return (x - mean) / tf.sqrt(variance + eps)


class DiffusionModel:
    def __init__(self, input_dim, t_embed_dim, hidden_dims, output_dim):
        """
        input_dim: Dimension of the flattened image (e.g. for 128x128x3, input_dim = 49152)
        t_embed_dim: Dimension for time embedding to be appended (e.g. 16)
        hidden_dims: List of hidden dimensions for each layer (e.g. [512, 1024, 512])
        output_dim: Dimension of the output (should equal input_dim)
        """
        # The first layer takes the concatenated [flattened image, time_embedding]
        self.W1 = tf.Variable(np.random.randn(input_dim + t_embed_dim, hidden_dims[0]) * 0.001, dtype=np.float32)
        self.b1 = tf.Variable(np.zeros((1, hidden_dims[0]), dtype=np.float32))
        
        self.W2 = tf.Variable(np.random.randn(hidden_dims[0], hidden_dims[1]) * 0.001, dtype=np.float32)
        self.b2 = tf.Variable(np.zeros((1, hidden_dims[1]), dtype=np.float32))
        
        self.W3 = tf.Variable(np.random.randn(hidden_dims[1], hidden_dims[2]) * 0.001, dtype=np.float32)
        self.b3 = tf.Variable(np.zeros((1, hidden_dims[2]), dtype=np.float32))
        
        self.W4 = tf.Variable(np.random.randn(hidden_dims[2], output_dim) * 0.001, dtype=np.float32)
        self.b4 = tf.Variable(np.zeros((1, output_dim), dtype=np.float32))
    
    def __call__(self, x_t, t):
        batch_size = tf.shape(x_t)[0]
        # Ensure the input is float32
        x_t = tf.cast(x_t, tf.float32)
        x_t_flat = tf.cast(tf.reshape(x_t, [batch_size, -1]), tf.float32)

        t_embed_dim = int(self.W1.shape[0]) - int(x_t_flat.shape[1])
        t_emb = tf.convert_to_tensor(get_t_embedding(t, t_embed_dim), dtype=tf.float32)
        t_emb = tf.tile(tf.expand_dims(t_emb, 0), [batch_size, 1])

        inp = tf.concat([x_t_flat, t_emb], axis=1)
        
        # Layer 1: Linear -> LayerNorm -> LeakyReLU
        h1 = tf.matmul(inp, self.W1) + self.b1
        h1 = layer_norm(h1)
        h1 = tf.nn.leaky_relu(h1, alpha=0.2)
        
        # Layer 2: Linear -> LayerNorm -> LeakyReLU
        h2 = tf.matmul(h1, self.W2) + self.b2
        h2 = layer_norm(h2)
        h2 = tf.nn.leaky_relu(h2, alpha=0.2)
        # Residual connection from h1 to h2 if dimensions match
        if h1.shape[-1] == h2.shape[-1]:
            h2 = h2 + h1
        
        # Layer 3: Linear -> LayerNorm -> LeakyReLU
        h3 = tf.matmul(h2, self.W3) + self.b3
        h3 = layer_norm(h3)
        h3 = tf.nn.leaky_relu(h3, alpha=0.2)
        # Residual connection from h2 to h3 if dimensions match
        if h2.shape[-1] == h3.shape[-1]:
            h3 = h3 + h2
        
        # Output layer (predict noise)
        output = tf.matmul(h3, self.W4) + self.b4
        return output
    

def get_batches(data, batch_size):
    np.random.shuffle(data)
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]


def train_diffusion_model(model, data, epochs, batch_size):
    optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    losses = []
    total_batches = int(np.ceil(len(data) / batch_size))
    
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        num_batches = 0
        avg_gradients = []
        print(f"Starting Epoch {epoch+1}/{epochs}", flush=True)
        
        # Wrap the batch generator with tqdm for a progress bar.
        for batch in tqdm(get_batches(data, batch_size), total=total_batches, desc=f"Epoch {epoch+1}/{epochs}"):
            t = np.random.randint(1, T)  # Random timestep
            noise = np.random.randn(*batch.shape).astype(np.float32)
            x_t = forward_diffusion(batch, t, noise)
            
            with tf.GradientTape() as tape:
                epsilon_pred = model(x_t, t)
                loss = tf.reduce_mean(tf.square(noise - epsilon_pred))
            
            grads = tape.gradient(loss, [model.W1, model.b1, model.W2, model.b2,
                                          model.W3, model.b3, model.W4, model.b4])
            optimizer.apply_gradients(zip(grads, [model.W1, model.b1, model.W2, model.b2,
                                                  model.W3, model.b3, model.W4, model.b4]))
            total_loss += loss.numpy()
            num_batches += 1
            avg_gradients.append([tf.reduce_mean(tf.abs(g)).numpy() for g in grads if g is not None])
        
        epoch_loss = total_loss / num_batches
        losses.append(epoch_loss)
        epoch_time = time.time() - start_time
        avg_grad_values = np.mean(avg_gradients, axis=0)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f} - Time: {epoch_time:.2f}s", flush=True)
        print(f"Average gradients: {avg_grad_values}", flush=True)
    
    print("Training completed!", flush=True)
    return losses


def sample(model, shape):
    """
    Generate new samples by performing the reverse diffusion process.
    shape: (batch_size, height, width, channels)
    """
    x = np.random.randn(*shape).astype(np.float32)  # Start with pure noise
    
    for t in reversed(range(1, T)):
        epsilon_pred = model(tf.convert_to_tensor(x), t)
        # Reshape prediction back to the image shape if necessary
        epsilon_pred = epsilon_pred.numpy().reshape(shape)
        
        sqrt_alpha_t = math.sqrt(alpha[t])
        beta_t = beta[t]
        sqrt_one_minus_alpha_bar_t = math.sqrt(1 - alpha_bar[t])
        mu = (1 / sqrt_alpha_t) * (x - (beta_t / sqrt_one_minus_alpha_bar_t) * epsilon_pred)
        
        if t > 1:
            noise = np.random.randn(*x.shape).astype(np.float32) * math.sqrt(beta_t)
        else:
            noise = 0
        x = mu + noise
    
    # Rescale from [-1,1] to [0,255]
    x = np.clip((x + 1) * 127.5, 0, 255).astype(np.uint8)
    return x


def compare_generated_vs_real(generated_samples, real_samples, num_images=5, cmap=None):
    """
    Display a side-by-side comparison of generated vs. real images.
    
    Parameters:
    - generated_samples: list or array of generated images.
    - real_samples: list or array of real images.
    - num_images: number of images to display from each.
    - cmap: colormap to use for imshow (e.g., 'gray' for grayscale images).
    """
    num_images = min(num_images, len(generated_samples), len(real_samples))
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
    
    for i in range(num_images):
        axes[0, i].imshow(real_samples[i], cmap=cmap)
        axes[0, i].axis("off")
        axes[1, i].imshow(generated_samples[i], cmap=cmap)
        axes[1, i].axis("off")
    
    axes[0, 0].set_title("Echte Afbeeldingen")
    axes[1, 0].set_title("Gegenereerde Afbeeldingen")
    plt.show()