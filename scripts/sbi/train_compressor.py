import argparse
import pickle
from functools import partial

import h5py
import haiku as hk
import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from chainconsumer import ChainConsumer
from jax.lib import xla_bridge
from numpyro import distributions as dist
from sbi_lens.normflow.models import AffineCoupling, ConditionalRealNVP
from sbi_lens.normflow.train_model import TrainModel
from tqdm import tqdm

print(xla_bridge.get_backend().platform)

tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--total_steps", type=int, default=150_000)
parser.add_argument(
    "--map_kind", type=str, default="gaussian"
)  # nbody_with_baryon_ia or gaussian or nbody
parser.add_argument("--loss", type=str, default="mse")

args = parser.parse_args()

if args.loss == "mse":
    loss_name = "train_compressor_mse"
elif args.loss == "vmim":
    loss_name = "train_compressor_vmim"

print("######## CONFIG ########")

sigma_e = 0.26
galaxy_density = 10 / 4
field_size = size = 10
field_npix = xsize = 80
nside = 512
reso = size * 60 / xsize
nbins = 1
dim = 6


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
path = "/gpfsdswork/dataset/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/perm_0000/projected_probes_maps_baryonified512.h5"
m_data = h5py.File(path, "r")
m_data = np.array(m_data["kg"][f"stage3_lensing{4}"]) + np.array(
    m_data["ia"][f"stage3_lensing{4}"]
)
proj = hp.projector.GnomonicProj(rot=[0, 0, 0], xsize=xsize, ysize=xsize, reso=reso)
m_data = proj.projmap(m_data, vec2pix_func=partial(hp.vec2pix, nside))
m_data = dist.Independent(
    dist.Normal(
        m_data,
        sigma_e / jnp.sqrt(galaxy_density * (field_size * 60 / field_npix) ** 2),
    ),
    2,
).sample(jax.random.PRNGKey(0), (1,))

params_name = [
    r"$\Omega_m$",
    r"$\sigma_8$",
    r"$w_0$",
    r"$h_0$",
    r"$n_s$",
    r"$\Omega_b$",
]

print("######## DATA AUGMENTATION ########")
tf.random.set_seed(1)

if args.map_kind == "nbody_with_baryon_ia":
    print("nbody w baryon and ia")

    def augmentation_noise(
        example, sigma_e=0.26, galaxy_density=27, field_size=5, field_npix=256
    ):
        x = example["map_nbody_w_baryon_ia"]
        x += tf.random.normal(
            shape=(field_npix, field_npix),
            stddev=sigma_e
            / jnp.sqrt(galaxy_density * (field_size * 60 / field_npix) ** 2),
        )

        return {"maps": x, "theta": example["theta"]}

elif args.map_kind == "nbody":
    print("nbody")

    def augmentation_noise(
        example, sigma_e=0.26, galaxy_density=27, field_size=5, field_npix=256
    ):
        x = example["map_nbody"]
        x += tf.random.normal(
            shape=(field_npix, field_npix),
            stddev=sigma_e
            / jnp.sqrt(galaxy_density * (field_size * 60 / field_npix) ** 2),
        )

        return {"maps": x, "theta": example["theta"]}

elif args.map_kind == "gaussian":
    print("gaussian")

    def augmentation_noise(
        example, sigma_e=0.26, galaxy_density=27, field_size=5, field_npix=256
    ):
        x = example["map_gaussian"]
        x += tf.random.normal(
            shape=(field_npix, field_npix),
            stddev=sigma_e
            / jnp.sqrt(galaxy_density * (field_size * 60 / field_npix) ** 2),
        )

        return {"maps": x, "theta": example["theta"]}


def augmentation_flip(example):
    x = tf.expand_dims(example["maps"], -1)
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return {"maps": x, "theta": example["theta"]}


def rescale_h(example):
    x = example["theta"]
    index_to_update = 3
    x = tf.tensor_scatter_nd_update(x, [[index_to_update]], [x[index_to_update] / 100])
    return {"maps": example["maps"], "theta": x}


def augmentation(example):
    return rescale_h(
        augmentation_flip(
            augmentation_noise(
                example=example,
                sigma_e=sigma_e,
                galaxy_density=galaxy_density,
                field_size=field_size,
                field_npix=field_npix,
            )
        )
    )


print("######## CREATE COMPRESSOR ########")

# nf
bijector_layers_compressor = [128] * 2

bijector_compressor = partial(
    AffineCoupling, layers=bijector_layers_compressor, activation=jax.nn.silu
)

NF_compressor = partial(ConditionalRealNVP, n_layers=4, bijector_fn=bijector_compressor)


# theta_bijector = tfb.Chain([
#     tfb.Scale(
#         jnp.array([
#             0.09285661,
#             0.23046516,
#             0.27378845,
#             4.458831,
#             0.04350383,
#             0.00650289
#         ])
#     ),
#     tfb.Shift(
#         -jnp.array([
#             2.9245374e-01,
#             8.2852399e-01,
#             -9.4738042e-01,
#             7.1496910e+01,
#             9.6793532e-01,
#             4.5024041e-02
#         ])
#     ),
# ])

# class Flow_nd_Compressor(hk.Module):
#     def __call__(self, y):
#         nvp = NF_compressor(dim)(y)
#         return tfd.TransformedDistribution(
#             nvp,
#             tfb.Chain([tfb.Invert(theta_bijector)])
#         )


class Flow_nd_Compressor(hk.Module):
    def __call__(self, y):
        nvp = NF_compressor(dim)(y)
        return nvp


nf = hk.without_apply_rng(
    hk.transform(lambda theta, y: Flow_nd_Compressor()(y).log_prob(theta).squeeze())
)


# compressor
class CompressorCNN2D(hk.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def __call__(self, x):
        net_x = hk.Conv2D(32, 3, 2)(x)
        net_x = jax.nn.leaky_relu(net_x)
        net_x = hk.Conv2D(64, 3, 2)(net_x)
        net_x = jax.nn.leaky_relu(net_x)
        net_x = hk.Conv2D(128, 3, 2)(net_x)
        net_x = jax.nn.leaky_relu(net_x)
        net_x = hk.AvgPool(16, 8, "SAME")(net_x)
        net_x = hk.Flatten()(net_x)

        net_x = hk.Linear(64)(net_x)
        net_x = jax.nn.leaky_relu(net_x)
        net_x = hk.Linear(self.output_dim)(net_x)

        return net_x.squeeze()


compressor = hk.transform_with_state(lambda y: CompressorCNN2D(dim)(y))
compressor_eval = hk.transform_with_state(lambda y: CompressorCNN2D(dim)(y))


print("######## TRAIN ########")

# init compressor
parameters_resnet, opt_state_resnet = compressor.init(
    jax.random.PRNGKey(0), y=0.5 * jnp.ones([1, field_npix, field_npix, nbins])
)
# init nf
params_nf = nf.init(
    jax.random.PRNGKey(0), theta=0.5 * jnp.ones([1, dim]), y=0.5 * jnp.ones([1, dim])
)

parameters_compressor = hk.data_structures.merge(parameters_resnet, params_nf)

del parameters_resnet, params_nf

# define optimizer
total_steps = args.total_steps - args.total_steps // 3
lr_scheduler = optax.piecewise_constant_schedule(
    init_value=0.0005,
    boundaries_and_scales={
        int(total_steps * 0.1): 0.7,
        int(total_steps * 0.2): 0.7,
        int(total_steps * 0.3): 0.7,
        int(total_steps * 0.4): 0.7,
        int(total_steps * 0.5): 0.7,
        int(total_steps * 0.6): 0.7,
        int(total_steps * 0.7): 0.7,
        int(total_steps * 0.8): 0.7,
        int(total_steps * 0.9): 0.7,
    },
)

optimizer_c = optax.adam(learning_rate=lr_scheduler)
opt_state_c = optimizer_c.init(parameters_compressor)

model_compressor = TrainModel(
    compressor=compressor,
    nf=nf,
    optimizer=optimizer_c,
    loss_name=loss_name,
)


# train dataset
ds_tr = tfds.load("CosmogridGridDataset/grid", split="train")

ds_tr = ds_tr.repeat()
ds_tr = ds_tr.shuffle(800)
ds_tr = ds_tr.map(augmentation)
ds_tr = ds_tr.batch(128)
ds_tr = ds_tr.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = iter(tfds.as_numpy(ds_tr))

# test dataset
ds_te = tfds.load("CosmogridGridDataset/grid", split="test")

ds_te = ds_te.repeat()
ds_te = ds_te.shuffle(200)
ds_te = ds_te.map(augmentation)
ds_te = ds_te.batch(128)
ds_te = ds_te.prefetch(tf.data.experimental.AUTOTUNE)
ds_test = iter(tfds.as_numpy(ds_te))

update = jax.jit(model_compressor.update)

store_loss = []
loss_train = []
loss_test = []
for batch in tqdm(range(1, args.total_steps + 1)):
    ex = next(ds_train)
    if not jnp.isnan(ex["maps"]).any():
        b_loss, parameters_compressor, opt_state_c, opt_state_resnet = update(
            model_params=parameters_compressor,
            opt_state=opt_state_c,
            theta=ex["theta"],
            x=ex["maps"],
            state_resnet=opt_state_resnet,
        )

        store_loss.append(b_loss)

        if jnp.isnan(b_loss):
            print("NaN Loss")
            break

    if batch % 2000 == 0:
        # save params
        with open(
            f"./save_params/{args.loss}/{args.map_kind}/params_nd_compressor_batch{batch}.pkl",
            "wb",
        ) as fp:
            pickle.dump(parameters_compressor, fp)

        with open(
            f"./save_params/{args.loss}/{args.map_kind}/opt_state_resnet_batch{batch}.pkl",
            "wb",
        ) as fp:
            pickle.dump(opt_state_resnet, fp)

        # save plot losses
        plt.figure()
        plt.plot(store_loss[1000:])
        plt.title("Batch Loss")
        plt.savefig(f"./fig/{args.loss}/{args.map_kind}/loss_compressor")

        ex_test = next(ds_test)

        b_loss_test, _, _, _ = update(
            model_params=parameters_compressor,
            opt_state=opt_state_c,
            theta=ex_test["theta"],
            x=ex_test["maps"],
            state_resnet=opt_state_resnet,
        )

        loss_train.append(b_loss)
        loss_test.append(b_loss_test)

        jnp.save(
            f"./save_params/{args.loss}/{args.map_kind}/loss_train.npy", loss_train
        )
        jnp.save(f"./save_params/{args.loss}/{args.map_kind}/loss_test.npy", loss_test)

        plt.figure()
        plt.plot(loss_train, label="train loss")
        plt.plot(loss_test, label="test loss")
        plt.legend()
        plt.title("Batch Loss")
        plt.savefig(f"./fig/{args.loss}/{args.map_kind}/loss_compressor_train_test")

        # save contour plot
        y, _ = compressor_eval.apply(
            parameters_compressor,
            opt_state_resnet,
            None,
            m_data.reshape([1, field_npix, field_npix, nbins]),
        )

        nvp_sample_nd = hk.transform(
            lambda x: Flow_nd_Compressor()(x).sample(100000, seed=hk.next_rng_key())
        )
        sample_nd = nvp_sample_nd.apply(
            parameters_compressor,
            rng=jax.random.PRNGKey(43),
            x=y * jnp.ones([100000, dim]),
        )
        idx = jnp.where(jnp.isnan(sample_nd))[0]
        sample_nd = jnp.delete(sample_nd, idx, axis=0)

        plt.figure()
        c = ChainConsumer()
        c.add_chain(sample_nd, parameters=params_name, name="SBI")
        fig = c.plotter.plot(figsize=1.2, truth=truth)

        plt.savefig(
            f"./fig/{args.loss}/{args.map_kind}/contour_plot_compressor_batch{batch}"
        )
