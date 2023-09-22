import jax
import jax.numpy as jnp


def linear_up(step, total_steps, min_lr=0.0, max_lr=1.0):
    """Linearly increase from min to max (and constant afterwards)."""
    out = min_lr + max_lr * (step / total_steps)
    return jnp.min(jnp.array([out, max_lr]))


def linear_down(step, total_steps, min_lr=0.0, max_lr=1.0):
    """Linearly decrease from max_lr to min_lr (and constant afterwards)."""
    out = max_lr - linear_up(step, total_steps, 0., max_lr - min_lr)
    return jnp.max(jnp.array([out, min_lr]))


def triangle_schedule(max_lr=0.01, total_steps=6000, end_lr=0.0):
    """A 'cyclical' learning rate schedule. Increase linearly to max_lr, then
    decrease linearly back to end_lr (and constant afterwards)."""
    midpoint = total_steps // 2
    def schedule(step):
        return jax.lax.cond(
            step < midpoint,
            lambda s: linear_up(s, midpoint, max_lr=max_lr),
            lambda s: linear_down(s - midpoint, midpoint, end_lr, max_lr),
            step)
    return schedule


def add_cooldown(schedule, cooldown_start, cooldown_length=None):
    """Wraps a schedule to add a linear cooldown."""
    if cooldown_length is None:
        cooldown_length = cooldown_start // 5
    def wrapped(step):
        return jax.lax.cond(
            step < cooldown_start,
            lambda s: schedule(s),
            lambda s: schedule(cooldown_start) * linear_down(
                s - cooldown_start, cooldown_length),
            step)
    return wrapped


def add_warmup(schedule, warmup_length, max_lr=None):
    """Wraps a schedule to add a linear warmup."""
    if max_lr is None:
        max_lr = schedule(0) * 10
    warmup = triangle_schedule(max_lr, warmup_length, schedule(warmup_length))
    def wrapped(step):
        return jax.lax.cond(step < warmup_length, warmup, schedule, step)
    return wrapped


# Schedules
###########


def constant(lr):
    return lambda step: lr


def constant_then_cooldown(lr, total_steps, cooldown_start=None):
    """By default, start cooldown at 90% of total steps."""
    if cooldown_start is None:
        cooldown_start = int(total_steps * 0.9)
    return add_cooldown(constant(lr), cooldown_start, total_steps - cooldown_start)


def constant_with_warmup_and_cooldown(
        lr, total_steps, warmup_length, cooldown_start, max_lr=None):
    if max_lr is None:
        max_lr = lr * 10
    schedule = constant_then_cooldown(lr, total_steps, cooldown_start)
    return add_warmup(schedule, warmup_length, max_lr=max_lr)