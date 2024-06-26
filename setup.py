from setuptools import setup

setup(
    name="learn2map",
    version="1.0",
    description=" Learning complex transformations through VAE",
    author="Justinezgh",
    packages=["learn2map"],
    install_requires=[
        "jax-cosmo",
        "numpyro",
        "lenstools",
        "sbi_lens @ git+https://github.com/DifferentiableUniverseInitiative/sbi_lens.git",
    ],
)
