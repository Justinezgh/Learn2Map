import math
import jax.numpy as jnp

def radial_profile(power_spectrum_2d):
    """
    Compute the radial profile of 2d image
    :param data: 2d image
    :return: radial profile
    """
    center = power_spectrum_2d.shape[0]/2
    v, u = jnp.indices((power_spectrum_2d.shape))
    k = jnp.sqrt((u - center)**2 + (v - center)**2)
    k = k.astype('int32')
    
    length = int(math.sqrt(2*(power_spectrum_2d.shape[0]/2)**2))+1
    tbin = jnp.bincount(k.ravel(), power_spectrum_2d.ravel(), length=length)
    nr = jnp.bincount(k.ravel(), length=length)  
    radialprofile = tbin / nr
    return radialprofile

def measure_power_spectrum(map_data, zero_freq_val=1e-2):
    """
    measures power 2d data
    :return: power spectrum (pk)
    """
    data_ft = jnp.abs(jnp.fft.fft2(map_data))
    data_ft_shifted = jnp.fft.fftshift(data_ft) 
    power_spectrum_2d = jnp.abs(data_ft_shifted * jnp.conjugate(data_ft_shifted)) / map_data.shape[0]**2
    nyquist = int(data_ft_shifted.shape[0] / 2)
    radialprofile = radial_profile(power_spectrum_2d)
    power_spectrum_1d = radialprofile[:nyquist]
    power_spectrum_1d = power_spectrum_1d.at[0].set(zero_freq_val)

    return power_spectrum_1d

def make_ps_map(power_spectrum, size, kps=None, zero_freq_val=1e7):
  #Ok we need to make a map of the power spectrum in Fourier space
  k1 = jnp.fft.fftfreq(size)
  k2 = jnp.fft.fftfreq(size)
  kcoords = jnp.meshgrid(k1,k2)
  # Now we can compute the k vector
  k = jnp.sqrt(kcoords[0]**2 + kcoords[1]**2)
  if kps is None:
    kps = jnp.linspace(0,0.5,len(power_spectrum))
  # And we can interpolate the PS at these positions
  ps_map = jnp.interp(k.flatten(), kps, power_spectrum).reshape([size,size])
  ps_map = ps_map.at[0,0].set(zero_freq_val)
  
  return ps_map # Carefull, this is not fftshifted


# from https://github.com/EiffL/LPTLensingComparison/blob/c4618d09b2b1df4b3b63799b304442f17bb709fd/jax_lensing/testf.py
def filter_cut(ngrid, reso_rad, cut_off):
    nsub = int(ngrid / 2 + 1)
    i, j = jnp.meshgrid(jnp.arange(nsub), jnp.arange(nsub))
    submatrix = 2 * jnp.pi * jnp.sqrt(i**2 + j**2) / reso_rad / jnp.float32(ngrid)

    result = jnp.zeros([ngrid, ngrid])
    result = result.at[0:nsub, 0:nsub].set(submatrix)
    result = result.at[0:nsub, nsub:].set(jnp.fliplr(submatrix[:, 1:-1]))
    result = result.at[nsub:, :].set(jnp.flipud(result[1:nsub-1, :]))
    tmp = jnp.around(result).astype(int)

    mask = jnp.ones_like(tmp)
    mask = mask.at[tmp > cut_off].set(0)
    return mask