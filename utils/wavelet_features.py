from jax.nn import standardize
import jax.numpy as jnp
from kymatio.jax import Scattering2D

def get_groupnorm_fun(n_groups, eps=1e-05, channel_dim=1):
  def groupnorm_fun(x):
    groups_list = jnp.array_split(x, n_groups, channel_dim)
    groups_list_normed = [standardize(k, channel_dim, epsilon=eps) for k in groups_list]
    return jnp.concatenate(groups_list_normed, channel_dim)
  return groupnorm_fun


def get_scatter_transform(image_hw=28, image_n_channels=1, j=2):
  scattering = Scattering2D(J=j, shape=(image_hw, image_hw))
  n_channels = 81 * image_n_channels
  out_hw = image_hw // 4
  return scattering, n_channels, out_hw

class MnistScatterEnc:
  def __init__(self, j, norm_features=False):
    scattering, n_channels, out_hw = get_scatter_transform(image_hw=28, image_n_channels=1, j=j)
    self.scatter = scattering
    self.groupnorm = get_groupnorm_fun(n_groups=27)
    self.norm_features = norm_features

  def __call__(self, x):
    bs = x.shape[0]
    x = jnp.reshape(x, (bs, 28, 28))
    x = self.scatter(x)
    x = self.groupnorm(x)
    x = jnp.reshape(x, (bs, -1))
    if self.norm_features:
      feat_norms = jnp.linalg.norm(x, axis=1, keepdims=True)
      x = x / feat_norms
    return x

class CifarScatterEnc:
  def __init__(self, j, norm_features=False):
    scattering, n_channels, out_hw = get_scatter_transform(image_hw=32, image_n_channels=3, j=j)
    self.scatter = scattering
    self.groupnorm = get_groupnorm_fun(n_groups=81, channel_dim=2)
    self.norm_features = norm_features

  def __call__(self, x):
    bs = x.shape[0]
    x = jnp.moveaxis(x, 3, 1)
    x = self.scatter(x)
    x = jnp.reshape(x, (bs, -1))
    if self.norm_features:
      feat_norms = jnp.linalg.norm(x, axis=1, keepdims=True)
      x = x / feat_norms
    return x

#def scatter_sanity_check():
#  import kymatio.torch as kymatio_torch
#  import numpy as np
#  image_hw = 28
#  bs = 10
#  x = np.random.randn(bs, image_hw, image_hw)
#  pt_scatter_fun = kymatio_torch.Scattering2D(J=2, shape=(image_hw, image_hw))
#  jax_scatter_fun = Scattering2D(J=2, shape=(image_hw, image_hw))
#  pt_sx = pt_scatter_fun(pt.tensor(x, dtype=pt.float32))
#  jax_sx = jax_scatter_fun(x)
#  pt_sx = pt_sx.numpy()
#  jax_sx = np.asarray(jax_sx)
#  print(np.linalg.norm(pt_sx - jax_sx))  # 2.8298905e-06
#  print(np.linalg.norm(pt_sx), np.linalg.norm(jax_sx))  # 15.964201 15.9642
  # passes!


if __name__ == '__main__':
  # scatter_sanity_check()
  pass

