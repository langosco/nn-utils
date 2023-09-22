import functools
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, random
import jax
import dm_pix as pix


# assume image is in [0, 1] and is of shape [H, W, C]


def random_crop(rng, image):
    """Randomly crop and resize an image."""
    shape = image.shape
    image = pix.random_crop(rng, image, shape - np.array([8, 8, 0]))
    image = jax.image.resize(image, shape, method="bicubic")
    return image


def maybe_apply(augmentation_fn):
    def wrapped(rng, img):
        subrng, rng = jax.random.split(rng)
        apply = random.bernoulli(subrng, 0.5)
        return jax.lax.cond(
            apply,
            lambda img: augmentation_fn(rng, img),
            lambda img: img,
            img
        )
    return wrapped


def augment_datapoint(rng, img):
    """Apply a random augmentation to a single image. Pixel values are assumed to be in [0, 1]"""
    rng = random.split(rng, 7)
    img = pix.random_flip_left_right(rng[0], img)
    img = maybe_apply(lambda _, img: pix.rot90(img))(rng[1], img)
    img = maybe_apply(random_crop)(rng[2], img)
    img = pix.random_contrast(rng[1], img, lower=0.3, upper=1.8)
    return img


def process_datapoint(rng: jnp.ndarray, 
                      img: jnp.array,
                      augment: bool = True) -> jnp.array:
    return jax.lax.cond(augment, 
                        lambda img: augment_datapoint(rng, img),
                        lambda img: img,
                        img)


@functools.partial(jit, static_argnames="augment")
def process_batch(rng, batch, augment=True):
    """Apply a random augmentation to a batch of images."""
    rng = random.split(rng, len(batch))
    proc = functools.partial(process_datapoint, augment=augment)
    return vmap(proc)(rng, batch)
