import tensorflow as tf
import time
from fid_eval import FIDEvaluator

GP_WEIGHT = 10.0

def gradient_penalty(batch_size, real_images, fake_images, labels, discriminator):
    """Calculates the gradient penalty for WGAN-GP."""
    # 1. Create random weights for interpolation
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
    
    # 2. Interpolate between real and fake images
    interpolated = real_images + alpha * (fake_images - real_images)

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 3. Get discriminator output for interpolated images
        pred = discriminator([interpolated, labels], training=True)

    # 4. Calculate gradients of predictions with respect to interpolated images
    grads = gp_tape.gradient(pred, [interpolated])[0]
    
    # 5. Calculate the norm of the gradients
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    
    # 6. Return the penalty: lambda * (||grad|| - 1)^2
    return tf.reduce_mean((norm - 1.0) ** 2)

@tf.function
def train_step_wgan(images, labels, generator, discriminator, gen_optimizer, disc_optimizer, latent_dim):
    batch_size = tf.shape(images)[0]
    noise = tf.random.normal([batch_size, latent_dim])

    # --- Train Discriminator (Critic) ---
    with tf.GradientTape() as disc_tape:
        fake_images = generator([noise, labels], training=True)
        
        fake_logits = discriminator([fake_images, labels], training=True)
        real_logits = discriminator([images, labels], training=True)

        # Wasserstein Loss: D(fake) - D(real)
        disc_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
        
        # Gradient Penalty
        gp = gradient_penalty(batch_size, images, fake_images, labels, discriminator)
        
        # Total Discriminator Loss
        disc_loss = disc_cost + gp * GP_WEIGHT

    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    # --- Train Generator ---
    with tf.GradientTape() as gen_tape:
        fake_images = generator([noise, labels], training=True)
        fake_logits = discriminator([fake_images, labels], training=True)
        
        # Generator wants Critic to output high values for fake images
        gen_loss = -tf.reduce_mean(fake_logits)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    
    return gen_loss, disc_loss

def train_cgan(dataset, generator, discriminator, epochs, latent_dim):
    # WGANs perform better with lower learning rates and 0.0 momentum (beta_1)
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)
    
    # Initialize our new FID Evaluator
    fid_eval = FIDEvaluator(input_shape=(75, 75, 3)) # Ensure channels match your dataset
    best_fid = float('inf')

    print("Starting WGAN-GP Training with FID Evaluation...")
    for epoch in range(epochs):
        start = time.time()
        gen_loss_avg = tf.keras.metrics.Mean()
        disc_loss_avg = tf.keras.metrics.Mean()

        for image_batch, label_batch in dataset:
            g_loss, d_loss = train_step_wgan(
                image_batch, label_batch, generator, discriminator, 
                gen_optimizer, disc_optimizer, latent_dim
            )
            gen_loss_avg.update_state(g_loss)
            disc_loss_avg.update_state(d_loss)

        print(f"Epoch {epoch + 1}/{epochs} | Time: {time.time()-start:.2f}s "
              f"| Gen Loss: {gen_loss_avg.result():.4f} | Disc Loss: {disc_loss_avg.result():.4f}")
        
        # --- Evaluate FID every 5 epochs ---
        if (epoch + 1) % 5 == 0:
            print("Calculating FID Score...")
            # Generate fake images for evaluation
            eval_noise = tf.random.normal([64, latent_dim])
            eval_labels = tf.random.uniform([64], minval=0, maxval=3, dtype=tf.int32)
            eval_fakes = generator([eval_noise, eval_labels], training=False)
            
            # Get a batch of real images
            real_images, _ = next(iter(dataset.take(1)))
            
            # Ensure we evaluate on the same amount of images
            eval_reals = real_images[:64]
            
            current_fid = fid_eval.get_fid_score(eval_reals, eval_fakes)
            print(f"---> Current FID Score: {current_fid:.4f}")
            
            if current_fid < best_fid:
                best_fid = current_fid
                generator.save('best_cgan_wgan_gp.keras')
                print(f"---> New Best Model Saved! (FID: {best_fid:.4f})")
        
    print("Training Finished!")