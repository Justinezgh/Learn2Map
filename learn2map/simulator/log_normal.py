from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.ndimage import map_coordinates

__all__ = ["lensingLogNormal"]

SOURCE_FILE = Path(__file__)
SOURCE_DIR = SOURCE_FILE.parent
ROOT_DIR = SOURCE_DIR.parent.resolve()
DATA_DIR = ROOT_DIR / "data"

lognormal_params = np.loadtxt(
    DATA_DIR / "lognormal_shift.csv", skiprows=1, delimiter=","
).reshape([8, 8, 3])


@jax.jit
def shift_fn(omega_m, sigma_8):
    omega_m = jnp.atleast_1d(omega_m)
    sigma_8 = jnp.atleast_1d(sigma_8)
    return map_coordinates(
        lognormal_params[:, :, 2],
        jnp.stack(
            [(omega_m - 0.2) / 0.2 * 8 - 0.5, (sigma_8 - 0.6) / 0.4 * 8 - 0.5], axis=0
        ).reshape([2, -1]),
        order=1,
        mode="nearest",
    ).squeeze()


# @jax.jit
def make_power_map(pk_fn, N, field_size, zero_freq_val=0.0):
    k = 2 * jnp.pi * jnp.fft.fftfreq(N, d=field_size / N)
    kcoords = jnp.meshgrid(k, k)
    k = jnp.sqrt(kcoords[0] ** 2 + kcoords[1] ** 2)
    ps_map = pk_fn(k)
    ps_map = ps_map.at[0, 0].set(zero_freq_val)
    return ps_map * (N / field_size) ** 2


# @jax.jit
def make_lognormal_power_map(power_map, shift, zero_freq_val=0.0):
    power_spectrum_for_lognorm = jnp.fft.ifft2(power_map).real
    power_spectrum_for_lognorm = jnp.log(1 + power_spectrum_for_lognorm / shift**2)
    power_spectrum_for_lognorm = jnp.abs(jnp.fft.fft2(power_spectrum_for_lognorm))
    power_spectrum_for_lognorm = power_spectrum_for_lognorm.at[0, 0].set(0.0)
    return power_spectrum_for_lognorm


# @jax.jit
def Pk_fn(k, cosmo, a_ai):
    pz = jc.redshift.smail_nz(0.5, 2.0, 1.0)
    tracer = jc.probes.WeakLensing([pz], ia_bias=a_ai)
    ell_tab = jnp.logspace(0, 4.5, 128)
    cell_tab = jc.angular_cl.angular_cl(cosmo, ell_tab, [tracer])[0]
    return jc.scipy.interpolate.interp(k.flatten(), ell_tab, cell_tab).reshape(k.shape)


def lensingLogNormal(
    N=128,  # number of pixels on the map
    map_size=5,  # map size in deg.
    gal_per_arcmin2=10,
    sigma_e=0.26,  # shape noise
    model_type="lognormal",  # either 'lognormal' or 'gaussian',
    with_ia=True,
):
    pix_area = (map_size * 60 / N) ** 2  # arcmin2
    map_size = map_size / 180 * jnp.pi  # radians

    # Sampling cosmology
    theta = numpyro.sample(
        "theta",
        dist.Independent(dist.Normal(jnp.array([0.3, 0.8]), 0.05 * jnp.ones(2)), 1),
    )
    # Sampling ia
    ia = numpyro.sample("ia", dist.Normal(0, 1))

    # Sampling latent variables
    z = numpyro.sample(
        "z", dist.MultivariateNormal(loc=jnp.zeros((N, N)), precision_matrix=jnp.eye(N))
    )

    cosmo = jc.Planck15(Omega_c=theta[0], sigma8=theta[1])

    if with_ia:
        ia_bias = jc.bias.constant_linear_bias(ia)

    else:
        ia_bias = None

    P = partial(Pk_fn, cosmo=cosmo, a_ai=ia_bias)
    power_map = make_power_map(P, N, map_size)

    if model_type == "lognormal":
        # Compute the shift parameter as a function of cosmology
        shift = shift_fn(cosmo.Omega_m, theta[1])
        power_map = make_lognormal_power_map(power_map, shift)

    # Convolving by the power spectrum
    field = jnp.fft.ifft2(jnp.fft.fft2(z) * jnp.sqrt(power_map)).real

    if model_type == "lognormal":
        field = shift * (jnp.exp(field - jnp.var(field) / 2) - 1)

    # Adding "observational noise"
    x = numpyro.sample(
        "x",
        dist.Independent(
            dist.Normal(field, sigma_e / jnp.sqrt(gal_per_arcmin2 * pix_area)), 2
        ),
    )
    return x
