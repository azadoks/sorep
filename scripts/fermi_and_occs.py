# %%
import pathlib as pl

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import sorep

# %%
sm = sorep.smearing.Delta(0.0, 1.0)

x = jnp.linspace(-10, 10, 1000)
y = sm.occupation_derivative(x)

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

axes[0].plot(x, sm.occupation(x))

axes[1].plot(x, sm.occupation_derivative(x))
axes[1].plot(x, -jax.vmap(jax.grad(sm.occupation))(x))

axes[2].plot(x, sm.occupation_2nd_derivative(x))
axes[2].plot(x, -jax.vmap(jax.hessian(sm.occupation))(x))

# %%
