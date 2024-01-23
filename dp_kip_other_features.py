import itertools
import time
import os
import gc
from absl import app
from absl import flags

import jax
from jax import grad
from jax import jit
from jax import random
from jax import vmap
from jax import scipy as sp
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp

import numpy as np
import numpy.random as npr

from autodp.calibrator_zoo import generalized_eps_delta_calibrator
from autodp.mechanism_zoo import SubsampleGaussianMechanism

import examples as datasets
from aux_files import class_balanced_sample, one_hot, get_tfds_dataset, get_normalization_data, normalize
from utils.wavelet_features import MnistScatterEnc, CifarScatterEnc
from encoder_pretraining.resnet import ResNetEnc

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_string('dataset', 'mnist', 'Supports mnist, fashion_mnist, svhn_cropped, cifar10, cifar100')
flags.DEFINE_float('learning_rate', .15, 'Learning rate for training')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('epochs', 100, 'Number of epochs')
flags.DEFINE_integer('seed', 0, 'Seed for jax PRNG')
flags.DEFINE_integer(
    'microbatches', None, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_float('kip_loss_reg', 1e-6, 'regularizer hyperparam for kip')
flags.DEFINE_string('model_dir', None, 'Model directory')
flags.DEFINE_integer('support_size', 10, 'Support dataset size')
flags.DEFINE_float('delta', 1e-5, 'Delta param for DP')
flags.DEFINE_float('epsilon', 1., 'Epsilon param for DP')
flags.DEFINE_string('feature_type', 'wavelet', 'supports perceptual and wavelet features')
flags.DEFINE_string('exp_name', 'temp', 'where to store logs')
flags.DEFINE_boolean('rand_init', False, 'If True, start with fully random support set')

# flags for wavelet features scatter_j
flags.DEFINE_integer('scatter_j', 2, 'number of scales for wavelet scattering')

# flags for perceptual features
flags.DEFINE_string('pretrain_dataset', 'cifar10', 'which data the encoder is trained on')
flags.DEFINE_boolean('pretrained_encoder', True, 'If False, encoder uses random init')

#flags for normalizing the features
flags.DEFINE_boolean('normalize_features', False, '')


def kip_loss(phi_support, y_support, phi_target, y_target, num_classes, reg=1e-6):
  k_ts = jnp.matmul(phi_target, jnp.transpose(phi_support)) / num_classes
  k_ss = jnp.matmul(phi_support, jnp.transpose(phi_support)) / num_classes

  k_ss_reg = (k_ss + jnp.abs(reg) * jnp.trace(k_ss) * jnp.eye(k_ss.shape[0]) / k_ss.shape[0])
  pred = jnp.dot(k_ts, sp.linalg.solve(k_ss_reg, y_support, assume_a='pos'))
  mse_loss = 0.5 * jnp.mean((pred - y_target) ** 2)
  return mse_loss

def forward_pass_kip_loss(x_support, batch, y_support, feature_extractor, num_classes, kip_loss_reg=1e-6):
  x_target, y_target = batch
  print("x_support.shape in forward_pass_kip_loss=", x_support.shape)
  print("x_test.shape in forward_pass_kip_loss=", x_target.shape)
  if len(x_support.shape) <= 2:
      x_support = jnp.reshape(x_support, (-1, 28, 28, 1))
  phi_support = feature_extractor(x_support)
  phi_target = feature_extractor(x_target)
  return kip_loss(phi_support, y_support, phi_target, y_target, num_classes, kip_loss_reg)

def eval_acc(x_support, y_support, x_test, y_test, feature_extractor, num_classes, kip_loss_reg=1e-6):
  print("x_support.shape in eval_acc=", x_support.shape)
  print("x_test.shape in eval_acc=", x_test.shape)
  if len(x_support.shape) <= 2:
      x_support = jnp.reshape(x_support, (-1, 28, 28, 1))
  if x_test.shape[0] > 10000:
    print("Reducing bs to fit in memory")
    mse_loss = 0.
    acc = 0.
    bs= 10000
    num_iters = int(x_test.shape[0] / bs)
    rest =  x_test.shape[0] - num_iters* bs
    #print("rest=", rest)
    phi_support = feature_extractor(x_support)
    for iter in range(num_iters):
      x_test_batch = x_test[iter * bs:(iter + 1) * bs]
      y_test_batch = y_test[iter * bs:(iter + 1) * bs]
      #print("x_test_batch.shape=", x_test_batch.shape)
      phi_target = feature_extractor(x_test_batch)

      k_ts = jnp.matmul(phi_target, jnp.transpose(phi_support)) / num_classes
      k_ss = jnp.matmul(phi_support, jnp.transpose(phi_support)) / num_classes

      k_ss_reg = (k_ss + jnp.abs(kip_loss_reg) * jnp.trace(k_ss) * jnp.eye(k_ss.shape[0]) / k_ss.shape[0])
      pred = jnp.dot(k_ts, sp.linalg.solve(k_ss_reg, y_support, assume_a='pos'))
      mse_loss += 0.5 * jnp.sum((pred - y_test_batch) ** 2)
      acc += jnp.sum(jnp.argmax(pred, axis=1) == jnp.argmax(y_test_batch, axis=1))

      del k_ts, k_ss, k_ss_reg, pred, phi_target
      gc.collect()
      
    if rest == 0:
      mse_loss = mse_loss /  x_test.shape[0]
      acc = acc /  x_test.shape[0]
    else:
      x_test_batch = x_test[num_iters * bs:]
      y_test_batch = y_test[num_iters * bs:]
      #print("last x_test_batch in eval_acc=", x_test_batch.shape)
      phi_target = feature_extractor(x_test_batch)
      k_ts = jnp.matmul(phi_target, jnp.transpose(phi_support)) / num_classes
      k_ss = jnp.matmul(phi_support, jnp.transpose(phi_support)) / num_classes

      k_ss_reg = (k_ss + jnp.abs(kip_loss_reg) * jnp.trace(k_ss) * jnp.eye(k_ss.shape[0]) / k_ss.shape[0])
      pred = jnp.dot(k_ts, sp.linalg.solve(k_ss_reg, y_support, assume_a='pos'))
      mse_loss += 0.5 * jnp.sum((pred - y_test_batch) ** 2)
      acc += jnp.sum(jnp.argmax(pred, axis=1) == jnp.argmax(y_test_batch, axis=1))

      mse_loss = mse_loss /  x_test.shape[0]
      acc = acc /  x_test.shape[0]

  else:  
    phi_support = feature_extractor(x_support)
    phi_target = feature_extractor(x_test)

    k_ts = jnp.matmul(phi_target, jnp.transpose(phi_support)) / num_classes
    k_ss = jnp.matmul(phi_support, jnp.transpose(phi_support)) / num_classes

    k_ss_reg = (k_ss + jnp.abs(kip_loss_reg) * jnp.trace(k_ss) * jnp.eye(k_ss.shape[0]) / k_ss.shape[0])
    pred = jnp.dot(k_ts, sp.linalg.solve(k_ss_reg, y_support, assume_a='pos'))
    mse_loss = 0.5 * jnp.mean((pred - y_test) ** 2)
    acc = jnp.mean(jnp.argmax(pred, axis=1) == jnp.argmax(y_test, axis=1))
  return mse_loss, acc

def clipped_grad(params, l2_norm_clip, single_example_batch, y_support, grad_fun):
  """Evaluate gradient for a single-example batch and clip its grad norm."""
  grads = grad_fun(params, single_example_batch, y_support)
  nonempty_grads, tree_def = tree_flatten(grads)

  total_grad_norm = jnp.linalg.norm(jnp.array([jnp.linalg.norm(neg.ravel()) for
                                               neg in nonempty_grads]))

  divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
  normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
  del grads, nonempty_grads, total_grad_norm, divisor
  return tree_unflatten(tree_def, normalized_nonempty_grads)

def private_grad(params, batch, rng, l2_norm_clip, noise_multiplier,
                 batch_size, y_support, grad_fun):
  """Return differentially private gradients for params, evaluated on batch."""
  grads_treedef = None
  clipped_grads = vmap(clipped_grad,
                         (None, None, 0, None, None))(params, l2_norm_clip, batch,
                                                      y_support, grad_fun)

  clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
  aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
  del clipped_grads, clipped_grads_flat

  rngs = random.split(rng, len(aggregated_clipped_grads))
  noised_aggregated_clipped_grads = [
      g + l2_norm_clip * noise_multiplier * random.normal(r, g.shape)
      for r, g in zip(rngs, aggregated_clipped_grads)]
  normalized_noised_aggregated_clipped_grads = [
      g / batch_size for g in noised_aggregated_clipped_grads]

  del aggregated_clipped_grads, noised_aggregated_clipped_grads, rngs

  return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads)

def shape_as_image(images, labels, dataset, dummy_dim=False):
  if dataset=='mnist' or dataset=='fashion_mnist':
    target_shape = (-1, 1, 28, 28, 1) if dummy_dim else (-1, 28, 28, 1)
    images_reshaped = jnp.reshape(images, target_shape)
  elif dataset=='cifar10' or dataset=='svhn_cropped' or dataset=='cifar100':
    target_shape = (-1, 1, 32, 32, 3) if dummy_dim else (-1, 32, 32, 3)
    images_reshaped = jnp.reshape(images, target_shape) 
  return images_reshaped, labels

def shape_as_image_only_imgs(images, labels, dataset, dummy_dim=False):
  if dataset=='mnist' or dataset=='fashion_mnist':
    target_shape = (-1, 1, 28, 28, 1) if dummy_dim else (-1, 28, 28, 1)
    images_reshaped = jnp.reshape(images, target_shape)
  elif dataset=='cifar10' or dataset=='svhn_cropped' or dataset=='cifar100':
    target_shape = (-1, 1, 32, 32, 3) if dummy_dim else (-1, 32, 32, 3)
    images_reshaped = jnp.reshape(images, target_shape) 
  return images_reshaped

def get_grad_fun(num_classes):

  if FLAGS.feature_type == 'wavelet':
    if (FLAGS.dataset == 'mnist') or (FLAGS.dataset == 'fashion_mnist'):
      scatter_net = MnistScatterEnc(j=FLAGS.scatter_j, norm_features=FLAGS.normalize_features)
    else:
      scatter_net = CifarScatterEnc(j=FLAGS.scatter_j, norm_features=FLAGS.normalize_features)


    def grad_fun(x_support, batch, y_support):
      return grad(forward_pass_kip_loss)(x_support, batch, y_support, scatter_net, num_classes,
                                         FLAGS.kip_loss_reg)

  elif FLAGS.feature_type == 'resnet':
    print("We are using perceptual features with pretrained_encoder={} and normalize_features={}".format(FLAGS.pretrained_encoder, FLAGS.normalize_features))
    enc = ResNetEnc(FLAGS.pretrain_dataset, pretrained=FLAGS.pretrained_encoder,
                    norm_features=FLAGS.normalize_features)

    def grad_fun(x_support, batch, y_support):
      return grad(forward_pass_kip_loss)(x_support, batch, y_support, enc, num_classes, FLAGS.kip_loss_reg)

  else:
    raise NotImplementedError(f'Unrecognized feature type {FLAGS.feature_type}')
  return grad_fun

def main(_):

  if FLAGS.dataset == 'cifar100':
    num_classes=100
  else:
    num_classes=10

  grad_fun = get_grad_fun(num_classes)

  if FLAGS.dataset == 'mnist':
    train_images, train_labels, test_images, test_labels = datasets.mnist()
    labels_train = jnp.argmax(train_labels, axis=1)
    y_train=one_hot(labels_train, num_classes)
  else:
    X_TRAIN_RAW, labels_train, X_TEST_RAW, LABELS_TEST = get_tfds_dataset(FLAGS.dataset)
    channel_means, channel_stds = get_normalization_data(X_TRAIN_RAW)
    train_images, test_images = normalize(X_TRAIN_RAW, channel_means, channel_stds), normalize(X_TEST_RAW, channel_means, channel_stds)
    y_train, test_labels = one_hot(labels_train, num_classes), one_hot(LABELS_TEST, num_classes) 

  print("train_images.shape=", train_images.shape)

  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, FLAGS.batch_size)
  num_batches = num_complete_batches + bool(leftover)
  key = random.PRNGKey(FLAGS.seed)

  if FLAGS.dpsgd:
    # code from: https://github.com/yuxiangw/autodp/blob/master/tutorials/tutorial_calibrator.ipynb
    dp_params = {}
    general_calibrate = generalized_eps_delta_calibrator()
    dp_params['sigma'] = None
    dp_params['coeff'] = FLAGS.epochs * num_train / FLAGS.batch_size
    dp_params['prob'] = FLAGS.batch_size / num_train
    general_calibrate(SubsampleGaussianMechanism, FLAGS.epsilon, FLAGS.delta, [0, 1000],
                      params=dp_params, para_name='sigma', name='Subsampled_Gaussian')
    print("SIGMA WITH AUTODP: ", dp_params['sigma'])

  def data_stream():
    rng = npr.RandomState(FLAGS.seed)
    while True:
      perm = rng.permutation(num_train)
      for batch_i in range(num_batches):
        batch_idx = perm[batch_i * FLAGS.batch_size:(batch_i + 1) * FLAGS.batch_size]
        yield train_images[batch_idx], y_train[batch_idx]

  batches = data_stream()
  opt_init, opt_update, get_params = optimizers.adam(FLAGS.learning_rate)

  @jit
  def update(_, i_, opt_state_, batch, y_support):
    params_ = get_params(opt_state_)
    return opt_update(i_, grad_fun(params_, batch, y_support), opt_state_)

  @jit
  def private_update(rng, i_, opt_state_, batch, y_support):
    params_ = get_params(opt_state_)
    rng = random.fold_in(rng, i_)  # get new key for new random numbers
    return opt_update(
        i_,
        private_grad(params_, batch, rng, FLAGS.l2_norm_clip,
                     dp_params['sigma'], FLAGS.batch_size, y_support,
                     grad_fun), opt_state_)

  """Initialize distilled images as random original ones"""
  _, labels_init, init_params, y_init = class_balanced_sample(FLAGS.support_size, labels_train,
                                                              train_images, y_train,
                                                              seed=FLAGS.seed)
  if FLAGS.rand_init:
      """Initialize distilled images as N(0,1)"""
    if FLAGS.dataset == 'mnist':
      init_params = random.normal(key, (FLAGS.support_size, train_images.shape[1]))
      init_params = shape_as_image_only_imgs(init_params, y_init, FLAGS.dataset)
    else:
      init_params = random.normal(key, (FLAGS.support_size, train_images.shape[1],  train_images.shape[2],  train_images.shape[3]))
  opt_state = opt_init(init_params)
  itercount = itertools.count()

  logreg_accs = []
  print('\nStarting training...')
  params = None
  for epoch in range(1, FLAGS.epochs + 1):
    start_time = time.time()
    for _ in range(num_batches):
      if FLAGS.dpsgd:
        gc.collect()
        opt_state = \
            private_update(
                key, next(itercount), opt_state,
                shape_as_image(*next(batches), FLAGS.dataset, dummy_dim=True), y_init)

      else:
        gc.collect()
        opt_state = update(
            key, next(itercount), opt_state, shape_as_image(*next(batches), FLAGS.dataset), y_init)

    epoch_time = time.time() - start_time
    print(f'Epoch {epoch} in {epoch_time:0.2f} sec')
    # evaluate test accuracy
    params = get_params(opt_state)
    print("params.shape=", params.shape)
    if FLAGS.feature_type == 'wavelet':
      if (FLAGS.dataset == 'mnist') or (FLAGS.dataset == 'fashion_mnist'):
        feature_extractor = MnistScatterEnc(j=FLAGS.scatter_j, norm_features=FLAGS.normalize_features)
      else:
        feature_extractor = CifarScatterEnc(j=FLAGS.scatter_j, norm_features=FLAGS.normalize_features)
    elif FLAGS.feature_type == 'resnet':
      print("We are using perceptual features with pretrained_encoder={} and normalize_features={}".format(FLAGS.pretrained_encoder, FLAGS.normalize_features))
      feature_extractor = ResNetEnc(FLAGS.pretrain_dataset, pretrained=FLAGS.pretrained_encoder,
                    norm_features=FLAGS.normalize_features)
    else:
      raise NotImplementedError(f'Unrecognized feature type {FLAGS.feature_type}')
    
    if len(test_images.shape) == 2:
      test_images, test_labels  = shape_as_image(test_images, test_labels, FLAGS.dataset)
    

    mse_loss, acc = eval_acc(params, y_init, test_images, test_labels, feature_extractor, num_classes, FLAGS.kip_loss_reg)
    print("test loss:", mse_loss)
    print("test acc: ", acc)
    
  cur_path = os.path.dirname(os.path.abspath(__file__))
  if FLAGS.feature_type == 'wavelet':
    save_acc_path=os.path.join(cur_path, 'accuracy_scatter')
    save_images_path=os.path.join(cur_path, 'images_scatter')
    if FLAGS.dpsgd:
      save_acc_path=os.path.join(cur_path, 'accuracy_scatter_dp')
      save_images_path=os.path.join(cur_path, 'images_scatter_dp')
  else:
    save_acc_path=os.path.join(cur_path, 'accuracy_perceptual')
    save_images_path=os.path.join(cur_path, 'images_perceptual')
    if FLAGS.dpsgd:
      save_acc_path=os.path.join(cur_path, 'accuracy_perceptual_dp')
      save_images_path=os.path.join(cur_path, 'images_perceptual_dp')

  if FLAGS.dpsgd:
    if FLAGS.feature_type == 'wavelet':
	    filename_acc="acc_{}_{}_supp_size={}_eps={}_delta={}_lr={}_c={}_bs={}_epochs={}_reg={}_seed={}_rand_init={}_scatter_j={}_norm_feat={}_rand_init={}.txt".format(FLAGS.feature_type, 
                    FLAGS.dataset,FLAGS.support_size, FLAGS.epsilon, FLAGS.delta, FLAGS.learning_rate, FLAGS.l2_norm_clip, FLAGS.batch_size, FLAGS.epochs, 
                    FLAGS.kip_loss_reg, FLAGS.seed, FLAGS.rand_init, FLAGS.scatter_j, FLAGS.normalize_features, FLAGS.rand_init)
    else:
	    filename_acc="acc_{}_{}_supp_size={}_eps={}_delta={}_lr={}_c={}_bs={}_epochs={}_reg={}_seed={}_pretrained_enc={}_norm_feat={}_rand_init={}.txt".format(FLAGS.feature_type, 
                    FLAGS.dataset,FLAGS.support_size, FLAGS.epsilon, FLAGS.delta, FLAGS.learning_rate, FLAGS.l2_norm_clip, FLAGS.batch_size, FLAGS.epochs, 
                    FLAGS.kip_loss_reg, FLAGS.seed, FLAGS.pretrained_encoder, FLAGS.normalize_features, FLAGS.rand_init)
  else:
    if FLAGS.feature_type == 'wavelet':
      filename_acc="acc_{}_{}_supp_size={}_lr={}_bs={}_epochs={}_reg={}_seed={}_rand_init={}_scatter_j={}_norm_feat={}.txt".format(FLAGS.feature_type, 
                    FLAGS.dataset,FLAGS.support_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.epochs, FLAGS.kip_loss_reg, FLAGS.seed, FLAGS.rand_init, 
                    FLAGS.scatter_j, FLAGS.normalize_features)
    else:
      filename_acc="acc_{}_{}_supp_size={}_lr={}_bs={}_epochs={}_reg={}_seed={}_pretrained_enc={}_norm_feat={}.txt".format(FLAGS.feature_type, 
                    FLAGS.dataset,FLAGS.support_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.epochs, FLAGS.kip_loss_reg, FLAGS.seed, 
                    FLAGS.pretrained_encoder, FLAGS.normalize_features)

  if FLAGS.dpsgd:
    if FLAGS.feature_type == 'wavelet':
	    filename_images="images_{}_{}_supp_size={}_eps={}_delta={}_lr={}_c={}_bs={}_epochs={}_reg={}_seed={}_rand_init={}_scatter_j={}_norm_feat={}_rand_init={}.npz".format(FLAGS.feature_type, 
                    FLAGS.dataset,FLAGS.support_size, FLAGS.epsilon, FLAGS.delta, FLAGS.learning_rate, FLAGS.l2_norm_clip, FLAGS.batch_size, FLAGS.epochs, 
                    FLAGS.kip_loss_reg, FLAGS.seed, FLAGS.rand_init, FLAGS.scatter_j, FLAGS.normalize_features, FLAGS.rand_init)
    else:
	    filename_images="images_{}_{}_supp_size={}_eps={}_delta={}_lr={}_c={}_bs={}_epochs={}_reg={}_seed={}_pretrained_enc={}_norm_feat={}_rand_init={}.npz".format(FLAGS.feature_type, 
                    FLAGS.dataset,FLAGS.support_size, FLAGS.epsilon, FLAGS.delta, FLAGS.learning_rate, FLAGS.l2_norm_clip, FLAGS.batch_size, FLAGS.epochs, 
                    FLAGS.kip_loss_reg, FLAGS.seed, FLAGS.pretrained_encoder, FLAGS.normalize_features, FLAGS.rand_init)
  else:
    if FLAGS.feature_type == 'wavelet':
      filename_images="images_{}_{}_supp_size={}_lr={}_bs={}_epochs={}_reg={}_seed={}_rand_init={}_scatter_j={}_norm_feat={}.npz".format(FLAGS.feature_type, 
                    FLAGS.dataset,FLAGS.support_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.epochs, FLAGS.kip_loss_reg, FLAGS.seed, FLAGS.rand_init, 
                    FLAGS.scatter_j, FLAGS.normalize_features)
    else:
      filename_images="images_{}_{}_supp_size={}_lr={}_bs={}_epochs={}_reg={}_seed={}_pretrained_enc={}_norm_feat={}.npz".format(FLAGS.feature_type, 
                    FLAGS.dataset,FLAGS.support_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.epochs, FLAGS.kip_loss_reg, FLAGS.seed, 
                    FLAGS.pretrained_encoder, FLAGS.normalize_features)


  if not os.path.exists(save_acc_path):
    os.makedirs(save_acc_path)
  with open(os.path.join(save_acc_path, filename_acc), 'w') as f:
    f.writelines(str(acc))


  params_final_x, params_init_raw_y=shape_as_image(params, labels_init, FLAGS.dataset)
  print("params.shape at the end of traininge=", params_final_x.shape)
  print("params_init_raw_y=", params_init_raw_y)

  if not os.path.exists(save_images_path):
    os.makedirs(save_images_path)
  np.savez(os.path.join(save_images_path, filename_images), data=params_final_x, labels=params_init_raw_y)


if __name__ == '__main__':
  app.run(main)



