from functools import partial

import h5py
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow_datasets.core.utils import gcs_utils

tfp = tfp.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors

# disable internet connection
gcs_utils.gcs_dataset_info_files = lambda *args, **kwargs: None
gcs_utils.is_dataset_on_gcs = lambda *args, **kwargs: False

_CITATION = """
"""

_DESCRIPTION = """
"""


class DatasetConfig(tfds.core.BuilderConfig):
    def __init__(
        self,
        *,
        xsize,
        size,
        **kwargs,
    ):
        v1 = tfds.core.Version("0.0.1")
        super().__init__(description=("Cosmogrid simulations."), version=v1, **kwargs)
        self.xsize = xsize
        self.size = size


class CosmogridGridDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Cosmogrid dataset."""

    VERSION = tfds.core.Version("0.0.1")
    RELEASE_NOTES = {
        "0.0.1": "Initial release.",
    }
    BUILDER_CONFIGS = [
        DatasetConfig(
            name="grid",
            xsize=80,
            size=10,
        ),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "map_nbody_w_baryon_ia": tfds.features.Tensor(
                        shape=[
                            self.builder_config.xsize,
                            self.builder_config.xsize,
                        ],
                        dtype=tf.float32,
                    ),
                    "map_nbody_w_baryon": tfds.features.Tensor(
                        shape=[
                            self.builder_config.xsize,
                            self.builder_config.xsize,
                        ],
                        dtype=tf.float32,
                    ),
                    "map_nbody_w_ia": tfds.features.Tensor(
                        shape=[
                            self.builder_config.xsize,
                            self.builder_config.xsize,
                        ],
                        dtype=tf.float32,
                    ),
                    "map_nbody": tfds.features.Tensor(
                        shape=[
                            self.builder_config.xsize,
                            self.builder_config.xsize,
                        ],
                        dtype=tf.float32,
                    ),
                    "map_gaussian": tfds.features.Tensor(
                        shape=[
                            self.builder_config.xsize,
                            self.builder_config.xsize,
                        ],
                        dtype=tf.float32,
                    ),
                    "theta": tfds.features.Tensor(shape=[6], dtype=tf.float32),
                }
            ),
            supervised_keys=None,
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "start": 1,
                    "end": 900,
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "start": 900,
                    "end": 1300,
                },
            ),
        ]

    def _generate_examples(self, start, end):
        """Yields examples."""

        filename = "/gpfsdswork/dataset/CosmoGridV1/CosmoGridV1_metainfo.h5"
        f = h5py.File(filename, "r")
        dataset_grid = f["parameters"]["grid"]

        nb_of_projected_map = 25
        cosmo_parameters = jnp.array(
            [
                dataset_grid["Om"],
                dataset_grid["s8"],
                dataset_grid["w0"],
                dataset_grid["H0"],
                dataset_grid["ns"],
                dataset_grid["Ob"],
            ]
        ).T

        nside = 512
        mean_pixel_area = 4 * np.pi / hp.nside2npix(nside)
        scaling_factor = 1 / mean_pixel_area
        xsize = self.builder_config.xsize  # width of figure in pixels
        size = self.builder_config.size  # Size of square in degrees
        reso = size * 60 / xsize
        master_key = jax.random.PRNGKey(0)

        for i in range(start, end):
            key, master_key, key2 = jax.random.split(master_key, 3)
            params = cosmo_parameters[i]
            path_string = "/gpfsdswork/dataset/" + dataset_grid["path_par"][i].decode(
                "utf-8"
            ).replace("CosmoGrid", "CosmoGridV1").replace("raw", "stage3_forecast")
            for j in range(7):
                filename = path_string + "perm_000" + str(j)
                filename_baryon = filename + "/projected_probes_maps_baryonified512.h5"
                filename_withouth_baryon = (
                    filename + "/projected_probes_maps_nobaryons512.h5"
                )
                sim_with_baryon = h5py.File(filename_baryon, "r")
                sim_without_baryon = h5py.File(filename_withouth_baryon, "r")

                # keeping only last tomo bins
                nbody_map = np.array(sim_without_baryon["kg"][f"stage3_lensing{4}"])
                nbody_map_with_ia = np.array(
                    sim_without_baryon["kg"][f"stage3_lensing{4}"]
                ) + np.array(sim_with_baryon["ia"][f"stage3_lensing{4}"])
                nbody_map_with_baryon_and_ia = np.array(
                    sim_with_baryon["kg"][f"stage3_lensing{4}"]
                ) + np.array(sim_with_baryon["ia"][f"stage3_lensing{4}"])
                nbody_map_with_baryon = np.array(
                    sim_with_baryon["kg"][f"stage3_lensing{4}"]
                )

                # building gaussian map
                lmax = 3 * nside - 1
                power_spectrum_nbody_map = hp.sphtfunc.anafast(nbody_map, lmax=lmax)
                z = np.random.randn(hp.nside2npix(nside)) * np.sqrt(scaling_factor)
                z = jax.random.normal(key2, (hp.nside2npix(nside),)) * np.sqrt(
                    scaling_factor
                )
                z = np.array(z)

                power_spectrum_noise = hp.sphtfunc.anafast(z, lmax=lmax)
                power_spectrum_target = power_spectrum_nbody_map / power_spectrum_noise
                alm_hp = hp.map2alm(z, lmax=lmax)
                alm = hp.sphtfunc.almxfl(alm_hp, np.sqrt(power_spectrum_target))
                gaussian_map = hp.alm2map(alm, nside, lmax=lmax)

                # projection
                key1, key2 = jax.random.split(key)
                lon = jax.random.randint(key1, (nb_of_projected_map,), -180, 180)
                lat = jax.random.randint(key2, (nb_of_projected_map,), -90, 90)
                for k in range(nb_of_projected_map):
                    proj = hp.projector.GnomonicProj(
                        rot=[lon[k], lat[k], 0], xsize=xsize, ysize=xsize, reso=reso
                    )
                    projection_nbody_map = proj.projmap(
                        nbody_map, vec2pix_func=partial(hp.vec2pix, nside)
                    )
                    projection_nbody_map_with_ia = proj.projmap(
                        nbody_map_with_ia, vec2pix_func=partial(hp.vec2pix, nside)
                    )
                    projection_nbody_map_with_baryon_and_ia = proj.projmap(
                        nbody_map_with_baryon_and_ia,
                        vec2pix_func=partial(hp.vec2pix, nside),
                    )
                    projection_nbody_map_with_baryon = proj.projmap(
                        nbody_map_with_baryon, vec2pix_func=partial(hp.vec2pix, nside)
                    )
                    projection_gaussian = proj.projmap(
                        gaussian_map, vec2pix_func=partial(hp.vec2pix, nside)
                    )
                    yield f"{i}-{j}-{k}", {
                        "map_nbody_w_baryon_ia": jnp.array(
                            projection_nbody_map_with_baryon_and_ia
                        ).squeeze(),
                        "map_nbody_w_baryon": jnp.array(
                            projection_nbody_map_with_baryon
                        ).squeeze(),
                        "map_nbody_w_ia": jnp.array(
                            projection_nbody_map_with_ia
                        ).squeeze(),
                        "map_nbody": jnp.array(projection_nbody_map).squeeze(),
                        "map_gaussian": jnp.array(projection_gaussian).squeeze(),
                        "theta": jnp.array(params),
                    }
