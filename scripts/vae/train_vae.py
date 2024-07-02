import argparse
import os
import pickle
from collections.abc import Mapping
from functools import partial

import astropy.units as u
import h5py
import haiku as hk
import healpy as hp
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from jax.lib import xla_bridge
from jax_cosmo.redshift import redshift_distribution
from lenstools import ConvergenceMap
from tqdm import tqdm
from unet_model import UResNet
from utils import measure_power_spectrum, make_ps_map

tfp = tfp.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors

print(xla_bridge.get_backend().platform)

np.float = float
np.complex = complex


# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--total_steps", type=int, default=50_000)
parser.add_argument("--lr_rate", type=float, default=1e-2)


args = parser.parse_args()


PATH_experiment = f"{args.total_steps}_{args.lr_rate}_new58"
os.makedirs(f"./fig/{PATH_experiment}")
os.makedirs(f"./save_params/{PATH_experiment}")


is_baryon = False
is_ia = False
print("######## CONFIG ########")

sigma_e = 0.26
galaxy_density = gal_per_arcmin2 = 10/4
field_size = map_size = size = 10
field_npix = N = xsize = 80
nside = 512
reso = size * 60 / xsize
nbins = 1
dim = 6
nside = 512
mean_pixel_area = 4 * np.pi / hp.nside2npix(nside)
scaling_factor = 1 / mean_pixel_area

# Create our fiducial observations
pix_area = (map_size * 60 / N) ** 2  # arcmin2
pixel_width = jnp.sqrt(pix_area)
map_size_rad = map_size / 180 * jnp.pi  # radians


print("######## OBSERVED DATA ########")
filename = "/gpfsdswork/dataset/CosmoGridV1/CosmoGridV1_metainfo.h5"
f = h5py.File(filename, "r")
dataset_grid = f["parameters"]["fiducial"]
cosmo_parameters = jnp.array(
    [
        dataset_grid["Om"],
        dataset_grid["s8"],
        dataset_grid["w0"],
        dataset_grid["H0"] / 100,
        dataset_grid["ns"],
        dataset_grid["Ob"],
    ]
).T
truth = list(cosmo_parameters[0])

if is_baryon:
    path = "/gpfsdswork/dataset/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/perm_0000/projected_probes_maps_baryonified512.h5"
else: 
    path = "/gpfsdswork/dataset/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/perm_0000/projected_probes_maps_nobaryons512.h5"

m_data = h5py.File(path, "r")
m_data = np.array(m_data["kg"][f"stage3_lensing{4}"])
if is_ia:
    m_data += np.array(m_data["ia"][f"stage3_lensing{4}"])
    
proj = hp.projector.GnomonicProj(rot=[0, 0, 0], xsize=xsize, ysize=xsize, reso=reso)
m_data_proj = proj.projmap(m_data, vec2pix_func=partial(hp.vec2pix, nside))
m_data_proj_noisy = dist.Independent(
    dist.Normal(
        m_data_proj,
        sigma_e / jnp.sqrt(galaxy_density * (field_size * 60 / field_npix) ** 2),
    ),
    2,
).sample(jax.random.PRNGKey(0), (1,))


print("######## DEFINING LATENT ANALYTICAL MODEL ########")


from jax_cosmo.redshift import redshift_distribution
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class smail_nz2(redshift_distribution):
    def pz_fn(self, z):
        a, b, z0 = self.params
        return z**a * jnp.exp(-((z / z0) ** b))*4


filename = "/gpfsdswork/dataset/CosmoGridV1/CosmoGridV1_metainfo.h5"
f = h5py.File(filename, "r")
dataset_grid = f["parameters"]["fiducial"]

cosmo_fid = jc.Planck15(
    Omega_c=dataset_grid["Om"][0] - dataset_grid["Ob"][0],
    Omega_b=dataset_grid["Ob"][0],
    h=dataset_grid["H0"][0] / 100,
    n_s=dataset_grid["ns"][0],
    sigma8=dataset_grid["s8"][0],
    Omega_k=0.0,
    w0=dataset_grid["w0"][0],
    wa=0.0,
)


def Pk_fn(k, cosmo, a_ai=None):
    pz = smail_nz2(3.53, 4.49, 1.03, gals_per_arcmin2=galaxy_density)
    tracer = jc.probes.WeakLensing([pz], ia_bias=a_ai)
    ell_tab = jnp.logspace(0, 4.5, 128)
    cell_tab = jc.angular_cl.angular_cl(cosmo, ell_tab, [tracer])[0]
    return jc.scipy.interpolate.interp(k.flatten(), ell_tab, cell_tab).reshape(k.shape)


def make_power_map(pk_fn, N, field_size, zero_freq_val=0.0):
    k = 2 * jnp.pi * jnp.fft.fftfreq(N, d=field_size / N)
    kcoords = jnp.meshgrid(k, k)
    k = jnp.sqrt(kcoords[0] ** 2 + kcoords[1] ** 2)
    ps_map = pk_fn(k)
    ps_map = ps_map.at[0, 0].set(zero_freq_val)
    return ps_map * (N / field_size) ** 2


P = partial(Pk_fn, cosmo=cosmo_fid, a_ai=None)

# Creating a power spectrum map
power_map = make_power_map(P, N, map_size_rad)
power_map = power_map.at[0, 0].set(1.0)


def log_gaussian_prior(map_data, ps_map=power_map, N=N):
    data_ft = jnp.fft.fft2(map_data).at[0, 0].set(0.0) / float(N)
    return -0.5 * jnp.sum(
        jnp.real(data_ft * jnp.conj(data_ft)) / (ps_map)
    )  # + jnp.log(jnp.sqrt(jnp.linalg.det(ps_map)))


print("######## TEST DATASET ########")
m_data = h5py.File(path, "r")
m_data = np.array(m_data["kg"][f"stage3_lensing{4}"])
if is_ia:
    m_data += np.array(m_data["ia"][f"stage3_lensing{4}"])
    
path_string = "/gpfsdswork/dataset/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/"
master_key = jax.random.PRNGKey(0)
dataset_test = []
nb_of_projected_map = 400
for i in range(3):
    key, master_key = jax.random.split(master_key)
    filename = path_string + f"perm_000{i}"
    if is_baryon:
        filename_baryon = filename + "/projected_probes_maps_baryonified512.h5"
    else: 
        filename_baryon = filename + "/projected_probes_maps_nobaryons512.h5"
    sim_with_baryon = h5py.File(filename_baryon, "r")
    nbody_map_with_baryon_and_ia = np.array(sim_with_baryon["kg"][f"stage3_lensing{4}"])
    if is_ia:
        nbody_map_with_baryon_and_ia += np.array(nbody_map_with_baryon_and_ia["ia"][f"stage3_lensing{4}"])
    # projection
    key1, key2 = jax.random.split(key)
    lon = jax.random.randint(key1, (nb_of_projected_map,), -180, 180)
    lat = jax.random.randint(key2, (nb_of_projected_map,), -90, 90)
    for k in range(nb_of_projected_map):
        proj = hp.projector.GnomonicProj(
            rot=[lon[k], lat[k], 0], xsize=xsize, ysize=xsize, reso=reso
        )
        projection_nbody = proj.projmap(
            nbody_map_with_baryon_and_ia, vec2pix_func=partial(hp.vec2pix, nside)
        )
        dataset_test.append(projection_nbody)

dataset_test = jnp.array(dataset_test).reshape([-1, field_npix, field_npix])


print("######## DATA AUGMENTATION ########")


def augmentation_noise(
    example, sigma_e=0.26, galaxy_density=27, field_size=5, field_npix=256
):
    x = example["map_nbody"]
    x += tf.random.normal(
        shape=(field_npix, field_npix),
        stddev=sigma_e / jnp.sqrt(galaxy_density * (field_size * 60 / field_npix) ** 2),
    )

    return {"maps": x, "theta": example["theta"]}


def augmentation_flip(example):
    x = tf.expand_dims(example["maps"], -1)
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return {"maps": x, "theta": example["theta"]}

def mean_to_zero(example):
    x = example["map_nbody"]
    x = x - tf.reduce_mean(x)
    return {"map_nbody": x, "theta": example["theta"]}

def rescale_h(example):
    x = example["theta"]
    index_to_update = 3
    x = tf.tensor_scatter_nd_update(x, [[index_to_update]], [x[index_to_update] / 100])
    return {"maps": example["maps"], "theta": x}

def augmentation(example):
    return rescale_h(
        augmentation_flip(
            augmentation_noise(
                example=mean_to_zero(example),
                sigma_e=sigma_e,
                galaxy_density=galaxy_density,
                field_size=field_size,
                field_npix=field_npix,
            )
        )
    )


print("######## CREATE VAE ########")

# Unet from Benjamin Remy
class UResNetEncoder(UResNet):
    """ResNet18."""

    def __init__(
        self,
        bn_config = None,
        use_bn = True,
        pad_crop = False,
        n_output_channels = 1,
        name    = None,
    ):
        """Constructs a ResNet model.
        Args:
          bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
            passed on to the :class:`~haiku.BatchNorm` layers.
          resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
            to ``False``.
          use_bn: Whether the network should use batch normalisation. Defaults to
            ``True``.
          n_output_channels: The number of output channels, for example to change in
            the case of a complex denoising. Defaults to 1.
          name: Name of the module.
        """
        super().__init__(
            blocks_per_group=(2, 2),
            bn_config=bn_config,
            bottleneck=False,
            channels_per_group=(4, 8),
            use_projection=(True, True),
            strides=(2, 1),
            use_bn=use_bn,
            pad_crop=pad_crop,
            n_output_channels=n_output_channels,
            name=name,
        )

# does not take into account channel + if several tomo bin we have 
# to change these fun to compute auto and cross power spectra
@jax.jit
@jax.vmap
def normalize(map1, map2): 
    map1 = map1.squeeze()
    map2 = map2.squeeze()
    ps_map1 = make_ps_map(measure_power_spectrum(map1), N)
    ps_map2 = make_ps_map(measure_power_spectrum(map2), N)
    cancel_ps = jnp.fft.fft2(map1) / (jnp.sqrt(ps_map1) + 1e-10)
    apply_ps_map = jnp.fft.ifft2(cancel_ps * jnp.sqrt(ps_map2)).real
    return apply_ps_map

class ConvDecoder(hk.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        
    def __call__(self, x):
        
        residual = hk.Conv2D(1, 3, 1)(x)
        residual = jax.nn.leaky_relu(residual)
        residual = hk.Conv2D(1, 3, 1)(residual) 
        residual = jax.nn.leaky_relu(residual)
        residual = hk.Conv2D(1, 3, 1)(residual)

        res = (residual + x)
        
        return normalize(res, x)


# define decoder and encoder
encoder = hk.without_apply_rng(
    hk.transform_with_state(
        lambda x: UResNetEncoder(n_output_channels=2, name="encoder")(
            x.reshape([-1, N, N, 1]), condition=None, is_training=True
        )
    )
)
encoder_eval = hk.without_apply_rng(
    hk.transform_with_state(
        lambda x: UResNetEncoder(n_output_channels=2, name="encoder")(
            x.reshape([-1, N, N, 1]), condition=None, is_training=False
        )
    )
)
params_encoder, state_encoder = encoder.init(
    jax.random.PRNGKey(0), jnp.ones([1, N, N, 1])
)


decoder = hk.without_apply_rng(
    hk.transform_with_state(
        lambda z: ConvDecoder(output_dim=1)(
            z.reshape([-1, N, N, 1])
        )
    )
)
decoder_eval = hk.without_apply_rng(
    hk.transform_with_state(
        lambda z: ConvDecoder(output_dim=1)(
            z.reshape([-1, N, N, 1])
        )
    )
)
params_decoder, state_decoder = decoder.init(
    jax.random.PRNGKey(0), jnp.ones([1, N, N, 1])
)

vae_params = hk.data_structures.merge(params_encoder, params_decoder)


print("######## ELBO LOSS FUNCTION ########")


@jax.jit
def posterior_z(y):
    return tfd.MultivariateNormalDiag(
        loc=y[..., 0],
        scale_diag=tfb.Softplus(low=1e-8).forward(y[..., 1] + 1e-3),
        
    )


@jax.jit
def posterior_x(y):
    return dist.Independent(
        dist.Normal(
            y,
            sigma_e / jnp.sqrt(galaxy_density * (field_size * 60 / field_npix) ** 2),
        ),
        2,
    )


@jax.jit
def log_posterior_x(y, x):
    return posterior_x(y).log_prob(x)

def loss_elbo(params, state, x, rng, weight_dkl, weight_nll):
    M = 20

    state_encoder, state_decoder = state

    print("checking dimensions")
    # Encode
    print("x.shape", x.shape)
    output_encoder, state_encoder = encoder.apply(
        params, state_encoder, x.reshape([-1, N, N, 1])
    )
    print("output_encoder.shape", output_encoder.shape)

    # Sample code
    p_z = posterior_z(output_encoder)
    z = p_z.sample(M, seed=rng)
    print("z.shape", z.shape)
    logp_z = p_z.log_prob(z).sum(-1)
    print("logp_z.shape", logp_z.shape)
    log_prior = jax.vmap(log_gaussian_prior)(z.reshape([-1, N, N])).reshape([M, -1])
    print("log_prior.shape", log_prior.shape)

    # Decode
    output_decoder, state_decoder = jax.vmap(
        decoder.apply,
        in_axes=(None, None, 0),
        out_axes=(0,None))(params, state_decoder, z)
    
    print("output_decoder.shape", output_decoder.shape)
    
    logp_x = jax.vmap(log_posterior_x, in_axes=(0, None))(output_decoder, x)
    print("logp_x.shape", logp_x.shape)

    elbo = (weight_nll * logp_x - weight_dkl * (logp_z - log_prior)).mean()
    print("loss shape", elbo.shape)

    return -elbo, ((state_encoder, state_decoder),
        (jnp.mean(-logp_x), jnp.mean(logp_z - log_prior)),
    )



print("######## TRAINING ########")


@jax.jit
def update(model_params, opt_state, state, x, rng, weight_dkl, weight_nll):
    (loss, state), grads = jax.value_and_grad(loss_elbo, has_aux=True)(
        model_params, state, x, rng, weight_dkl, weight_nll
    )
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(model_params, updates)
    return loss, new_params, new_opt_state, state[0], state[1]


# total_steps = 15_000
# lr_scheduler = optax.piecewise_constant_schedule(
#     init_value=args.lr_rate,
#     boundaries_and_scales={
#         int(total_steps * 0.1): 0.6,
#         int(total_steps * 0.3): 0.5,
#         int(total_steps * 0.5): 0.4,
#         int(total_steps * 0.7): 0.3,
#         int(total_steps * 0.9): 0.1,
#     },
# )
lr_scheduler = optax.exponential_decay(
    init_value=0.001,
    transition_steps=2_000,
    decay_rate=0.9,
    end_value=1e-5,
)

lr_scheduler_dkl = optax.exponential_decay(
    init_value=0.9,
    transition_steps=1_000,
    decay_rate=0.9,
    end_value=1e-6
)

lr_scheduler_nll = optax.exponential_decay(
    init_value=0.9,
    transition_steps=1_000,
    decay_rate=0.9,
    end_value=1e-6
)


if is_ia and is_baryon: 
    ds_tr = tfds.load("CosmogridGridFiducialDataset/fiducial", split="train")
if (not is_ia) and (not is_baryon): 
    ds_tr = tfds.load("CosmogridGridFiducialDataset/fiducial_nbody", split="train")

ds_tr = ds_tr.repeat()
ds_tr = ds_tr.shuffle(800)
ds_tr = ds_tr.map(augmentation)
ds_tr = ds_tr.batch(256)
ds_tr = ds_tr.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = iter(tfds.as_numpy(ds_tr))

optimizer = optax.adam(learning_rate=lr_scheduler)
opt_state = optimizer.init(vae_params)

state = (state_encoder, state_decoder)

store_loss = []
store_logp_z = []
store_logp_x = []
store_loss_test = []
store_loss_train = []
store_logp_z_test = []
store_logp_z_train = []
store_logp_x_test = []
store_logp_x_train = []
master_seed = jax.random.PRNGKey(0)


for batch in tqdm(range(1, args.total_steps)):
    master_seed, rng = jax.random.split(master_seed, 2)
    ex = next(ds_train)
    x = ex["maps"].squeeze()
    weight_dkl = 1#1 - lr_scheduler_dkl(batch)
    weight_nll = 1 #- lr_scheduler_nll(batch)
    b_loss, vae_params, opt_state, state, logp = update(
        vae_params, opt_state, state, x, rng, weight_dkl, weight_nll
    )
    
    if jnp.isnan(b_loss):
            print("NaN Loss")
            break
            
    store_loss.append(b_loss)
    store_logp_z.append(logp[1])
    store_logp_x.append(logp[0])

    if batch % 2000 == 0:
        # save params
        with open(
            f"./save_params/{PATH_experiment}/params_nd_vae_batch{batch}.pkl", "wb"
        ) as fp:
            pickle.dump(vae_params, fp)

        with open(
            f"./save_params/{PATH_experiment}/opt_state_vae_batch{batch}.pkl", "wb"
        ) as fp:
            pickle.dump(opt_state, fp)

        with open(
            f"./save_params/{PATH_experiment}/state_vae_batch{batch}.pkl", "wb"
        ) as fp:
            pickle.dump(state, fp)

        # save plot losses
        plt.figure()
        plt.plot(store_loss[1000:])
        plt.title("Batch Loss")
        plt.savefig(f"./fig/{PATH_experiment}/loss_vae")

        # plt.figure()
        # plt.plot(jnp.mean(jnp.array(store_logp_z[1000:]),axis =1))
        # plt.title("logp_z")
        # plt.savefig(f"./fig/{PATH_experiment}/loss_dkl")

        # plt.figure()
        # plt.plot(jnp.mean(jnp.array(store_logp_x[1000:]),axis =1))
        # plt.title("logp_x")
        # plt.savefig(f"./fig/{PATH_experiment}/loss_likelihood")

        plt.figure()
        plt.plot(store_loss[int(batch - 2000) :])
        plt.title("Zoom batch Loss")
        plt.savefig(f"./fig/{PATH_experiment}/zoomloss_vae")

        # plt.figure()
        # plt.plot(jnp.mean(jnp.array(store_logp_z[int(batch - 2000) :]), axis = 1))
        # plt.title("zoom logp_z")
        # plt.savefig(f"./fig/{PATH_experiment}/zoomloss_dkl")

        # plt.figure()
        # plt.plot(jnp.mean(jnp.array(store_logp_x[int(batch - 2000) :]), axis = 1))
        # plt.title("zoom logp_x")
        # plt.savefig(f"./fig/{PATH_experiment}/zoomloss_likelihood")

        # check overfitting
        inds = np.random.randint(0, len(dataset_test), 128)
        weight_dkl = 1#1 - lr_scheduler_dkl(batch)
        weight_nll = 1 #- lr_scheduler_nll(batch)
        b_loss_test, _, _, _, logp_test = update(
            vae_params, opt_state, state, dataset_test[inds], rng, weight_dkl, weight_nll
        )
        store_loss_test.append(b_loss_test)
        store_logp_z_test.append(logp_test[1])
        store_logp_x_test.append(logp_test[0])
        store_loss_train.append(b_loss)
        store_logp_z_train.append(logp[1])
        store_logp_x_train.append(logp[0])

        # save plot losses
        plt.figure()
        plt.plot(store_loss_test, label="train")
        plt.plot(store_loss_train, label="train")
        plt.title("Batch Loss")
        plt.savefig(f"./fig/{PATH_experiment}/loss_vae_train_test")

        # check learning
        out_encoder, _ = encoder_eval.apply(vae_params, state[0], m_data_proj_noisy[0])
        sample_encoder = posterior_z(out_encoder.squeeze()).sample(
            seed=jax.random.PRNGKey(0)
        )
        out_vae, _ = decoder_eval.apply(vae_params, state[1], sample_encoder)
        sample_vae = posterior_x(out_vae.squeeze()).sample(jax.random.PRNGKey(10))

        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(out_vae.squeeze())
        plt.title("x prediction from vae")
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(m_data_proj.squeeze())
        plt.title("x")
        plt.colorbar()
        plt.subplot(133)
        plt.imshow(m_data_proj.squeeze() - out_vae.squeeze())
        plt.title("residuals")
        plt.colorbar()
        plt.savefig(f"./fig/{PATH_experiment}/learning_without_noise_{batch}")

        l_edges_kmap = np.arange(60.0, 2000.0, 15.0)
        kmap_lt_predicted = ConvergenceMap(out_vae.squeeze(), angle=map_size * u.deg)
        l1, Pl1 = kmap_lt_predicted.powerSpectrum(l_edges_kmap)

        kmap_lt_true = ConvergenceMap(m_data_proj.squeeze(), angle=map_size * u.deg)
        l2, Pl2 = kmap_lt_true.powerSpectrum(l_edges_kmap)
        
        plt.figure()
        plt.loglog(l1, Pl1, label="Predicted power spectrum")
        plt.loglog(l2, Pl2, "--", label="True power spectrum")
        plt.legend()
        plt.savefig(f"./fig/{PATH_experiment}/powerspectrum_learning_without_noise_{batch}")

        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(sample_vae.squeeze())
        plt.title("x prediction from vae")
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(m_data_proj_noisy.squeeze())
        plt.title("x")
        plt.colorbar()
        plt.subplot(133)
        plt.imshow(m_data_proj_noisy.squeeze() - sample_vae.squeeze())
        plt.title("residuals")
        plt.colorbar()
        plt.savefig(f"./fig/{PATH_experiment}/learning_with_noise_{batch}")

        l_edges_kmap = np.arange(60.0, 2000.0, 15.0)
        kmap_lt_predicted = ConvergenceMap(sample_vae.squeeze(), angle=map_size * u.deg)
        l1, Pl1 = kmap_lt_predicted.powerSpectrum(l_edges_kmap)

        kmap_lt_true = ConvergenceMap(
            m_data_proj_noisy.squeeze(), angle=map_size * u.deg
        )
        l2, Pl2 = kmap_lt_true.powerSpectrum(l_edges_kmap)
        
        plt.figure()
        plt.loglog(l1, Pl1, label="Predicted power spectrum")
        plt.loglog(l2, Pl2, "--", label="True power spectrum")
        plt.legend()
        plt.savefig(f"./fig/{PATH_experiment}/powerspectrum_learning_with_noise_{batch}")


# save params
with open(
    f"./save_params/{PATH_experiment}/params_nd_vae_batch{batch}.pkl", "wb"
) as fp:
    pickle.dump(vae_params, fp)

with open(
    f"./save_params/{PATH_experiment}/opt_state_vae_batch{batch}.pkl", "wb"
) as fp:
    pickle.dump(opt_state, fp)

with open(f"./save_params/{PATH_experiment}/state_vae_batch{batch}.pkl", "wb") as fp:
    pickle.dump(state, fp)