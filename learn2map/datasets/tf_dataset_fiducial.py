import os
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
        baryon,
        ia,
        **kwargs,
    ):
        v1 = tfds.core.Version("0.0.1")
        super().__init__(description=("Cosmogrid simulations."), version=v1, **kwargs)
        self.xsize = xsize
        self.size = size
        self.baryon = baryon
        self.ia = ia


class CosmogridGridFiducialDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Cosmogrid dataset."""

    VERSION = tfds.core.Version("0.0.1")
    RELEASE_NOTES = {
        "0.0.1": "Initial release.",
    }
    BUILDER_CONFIGS = [
        DatasetConfig(
            name="fiducial",  # nbody and baryon and ia
            xsize=80,
            size=10,
            baryon=True,
            ia=True,
        ),
        DatasetConfig(
            name="fiducial_nbody",  # nbody without baryon and ia
            xsize=80,
            size=10,
            baryon=False,
            ia=False,
        ),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "map_nbody": tfds.features.Tensor(
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
            tfds.core.SplitGenerator(name=tfds.Split.TRAIN),
        ]

    def _generate_examples(self):
        """Yields examples."""

        filename = "/gpfsdswork/dataset/CosmoGridV1/CosmoGridV1_metainfo.h5"
        f = h5py.File(filename, "r")
        dataset_grid = f["parameters"]["fiducial"]

        nb_of_projected_map = 600
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
        1 / mean_pixel_area
        xsize = self.builder_config.xsize  # width of figure in pixels
        size = self.builder_config.size  # Size of square in degrees
        reso = size * 60 / xsize
        master_key = jax.random.PRNGKey(0)

        path_string = (
            "/gpfsdswork/dataset/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/"
        )
        params = cosmo_parameters[0]

        for item in os.listdir(path_string):
            key, master_key = jax.random.split(master_key)
            # Check if the item is a directory
            if (
                (os.path.isdir(os.path.join(path_string, item)))
                and (item != "perm_0000")
                and (item != "perm_0001")
                and (item != "perm_0002")
            ):
                filename = path_string + item
                if self.builder_config.baryon:
                    filename_simulation = (
                        filename + "/projected_probes_maps_baryonified512.h5"
                    )
                else:
                    filename_simulation = (
                        filename + "/projected_probes_maps_nobaryons512.h5"
                    )

                simulation = h5py.File(filename_simulation, "r")
                # keeping only last tomo bins
                nbody_map = np.array(simulation["kg"][f"stage3_lensing{4}"])

                if self.builder_config.ia:
                    nbody_map += np.array(simulation["ia"][f"stage3_lensing{4}"])

                # projection
                key1, key2 = jax.random.split(key)
                lon = jax.random.randint(key1, (nb_of_projected_map,), -180, 180)
                lat = jax.random.randint(key2, (nb_of_projected_map,), -90, 90)
                for k in range(nb_of_projected_map):
                    proj = hp.projector.GnomonicProj(
                        rot=[lon[k], lat[k], 0], xsize=xsize, ysize=xsize, reso=reso
                    )
                    projection_nbody = proj.projmap(
                        nbody_map, vec2pix_func=partial(hp.vec2pix, nside)
                    )
                    yield f"{filename}-{k}", {
                        "map_nbody": jnp.array(projection_nbody).squeeze(),
                        "theta": jnp.array(params),
                    }
