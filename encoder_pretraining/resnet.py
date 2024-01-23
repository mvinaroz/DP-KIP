import os.path
from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional
import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.training import checkpoints

_ModuleDef = Any
FEAT_COL_NAME = 'intermediates'


def sow_replace(_xs, x):
  return x,

class _ResNetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: _ModuleDef
  norm: _ModuleDef
  act: Callable
  block_id: int
  strides: Tuple[int, int] = (1, 1)


  @nn.compact
  def __call__(self, x,):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    self.sow(FEAT_COL_NAME, f'b{self.block_id}_conv_0', x, reduce_fn=sow_replace)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    self.sow(FEAT_COL_NAME, f'b{self.block_id}_conv_1', x, reduce_fn=sow_replace)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class _BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  conv: _ModuleDef
  norm: _ModuleDef
  act: Callable
  block_id: int
  strides: Tuple[int, int] = (1, 1)


  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    self.sow(FEAT_COL_NAME, f'b{self.block_id}_conv_0', x, reduce_fn=sow_replace)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    self.sow(FEAT_COL_NAME, f'b{self.block_id}_conv_1', x, reduce_fn=sow_replace)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1))(y)
    self.sow(FEAT_COL_NAME, f'b{self.block_id}_conv_2', x, reduce_fn=sow_replace)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters * 4, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)

class _ResNet(nn.Module):
  """ResNetV1."""
  stage_sizes: Sequence[int]
  block_cls: _ModuleDef
  num_classes: int
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: _ModuleDef = nn.Conv
  dense_layer: bool = True

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = partial(self.conv, use_bias=False, dtype=self.dtype)

    norm = partial(nn.GroupNorm, dtype=self.dtype)

    x = conv(self.num_filters, (7, 7), (2, 2),
             padding=[(3, 3), (3, 3)],
             name='conv_init')(x)
    self.sow(FEAT_COL_NAME, 'first_conv', x, reduce_fn=sow_replace)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    block_id = 0
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(self.num_filters * 2 ** i,
                           strides=strides,
                           conv=conv,
                           norm=norm,
                           act=self.act,
                           block_id=block_id)(x)
        block_id += 1
    x = jnp.mean(x, axis=(1, 2))
    self.sow(FEAT_COL_NAME, 'flattened', x, reduce_fn=sow_replace)
    if self.dense_layer:
      x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x

_ResNet18 = partial(_ResNet, stage_sizes=[2, 2, 2, 2],
                    block_cls=_ResNetBlock)


def get_feats(model, variables, batch, n_blocks=8, n_block_convs=2):
  _, collections = model.apply(variables, batch, mutable=[FEAT_COL_NAME])
  feats = collections[FEAT_COL_NAME]
  feats_list = [feats['first_conv'][0]]

  for block_id in range(n_blocks):
    for conv_id in range(n_block_convs):
      conv_feats = feats[f'_ResNetBlock_{block_id}'][f'b{block_id}_conv_{conv_id}'][0]
      feats_list.append(conv_feats)

  feats_list.append(feats['flattened'][0])
  feats_list_flat = [jnp.reshape(k, (k.shape[0], -1)) for k in feats_list]
  return jnp.concatenate(feats_list_flat, axis=1)

def feat_extraction_test():
  model = _ResNet18(num_filters=64, num_classes=2)
  batch = jnp.ones((1, 28, 28, 3))
  variables = model.init(jax.random.PRNGKey(0), batch)
  feats_vec = get_feats(model, variables, batch)
  print(feats_vec.shape)

class ResNetEnc:
  def __init__(self, dataset, ckpt_dir='encoder_pretraining/selected_checkpoints/',
               pretrained=False, norm_features=False):
    self.model = _ResNet18(num_filters=64, num_classes=10, dense_layer=False)
    dummy_batch = jnp.ones((1, 28, 28, 3))
    self.variables = self.model.init(jax.random.PRNGKey(0), dummy_batch)
    self.norm_features = norm_features
    if pretrained:
      enc_checkpoints = {'cifar10': 'ckpt_cifar10',
                         'svhn': 'ckpt_svhn'}

      ckpt = checkpoints.restore_checkpoint(os.path.join(ckpt_dir, enc_checkpoints[dataset]),
                                            target=self.variables)
      self.variables = ckpt

  def __call__(self, x):
    x = jnp.repeat(x, 3, axis=3, total_repeat_length=3)
    feats_vec = get_feats(self.model, self.variables, x)
    if self.norm_features:
      feat_norms = jnp.linalg.norm(feats_vec, axis=1, keepdims=True)
      feats_vec = feats_vec / feat_norms
    return feats_vec


if __name__ == '__main__':
  feat_extraction_test()
