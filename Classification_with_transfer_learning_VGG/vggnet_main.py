#
#
#
import sys, os
import tensorflow as tf
from tuned_vgg16.Finetunning_vgg16 import tuned_vgg16
#from mynet_testing import test_cnn_net
#
cwd = os.getcwd()
sys.path.append(os.path.realpath(cwd))
#sys.path.append(cwd + '/models/VGG16')
tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.app.flags
#
# parameter setting with flags
flags.DEFINE_bool('train', True, 'run training')
flags.DEFINE_bool('test', False, 'run testing')
#
flags.DEFINE_integer('num_classes', 2, 'number of classes')
flags.DEFINE_integer('channels', 3, 'number of channels')
flags.DEFINE_integer('epochs', 200, 'number of epochs')
flags.DEFINE_integer('num_steps', 80, 'number of iteration steps')
flags.DEFINE_integer('num_tests', 50, 'number of testing iamges (maxumam number is 81)')
flags.DEFINE_integer('train_batch', 10, 'batch size of training data')
flags.DEFINE_integer('test_batch', 1, 'batch size for testin data')
flags.DEFINE_integer('valid_batch', 10, 'batch size of validation data')
flags.DEFINE_float('learning_rate', .0001, 'initial learning rate')
flags.DEFINE_integer('showing_step', 10, 'after how many steps to show results')
flags.DEFINE_integer('model_saving_step', 2000, 'after how many steps to save results automatically')
flags.DEFINE_integer('n_threads', 1, 'number of threads')
flags.DEFINE_bool('tensorboard', False, 'whether to load tensorboard in testing')
#
flags.DEFINE_string('log_path', cwd + '/tuned_vgg16/vgglog/', 'path to my models log (training)')
flags.DEFINE_string('vgg_ckpt_path', cwd + '/tuned_vgg16/chkpnt/vgg_16_2016_08_28/vgg_16.ckpt', 'path to vgg model checkpoint')
flags.DEFINE_string('chkpnt_path', cwd + '/tuned_vgg16/chkpnt/vgg16', 'path to vgg mata and checkpoints')

flags.DEFINE_string('training_path', cwd + '/making_data/training_susand_01.tfrecords', 'path to training data')
flags.DEFINE_string('validate_path', cwd + '/making_data/validate_susand_01.tfrecords', 'path to my validation data')
flags.DEFINE_string('testing_path', cwd + '/making_data/testing_susand_01.tfrecords', 'path to my testing data')
#
FLAGS = flags.FLAGS
#
# loading training function on my network ----
vgg16 = tuned_vgg16(FLAGS)

if FLAGS.train:
    vgg16.train_and_validate_vgg()

# testing my network ----
if FLAGS.test:
    vgg16.test_vgg()
