# coding=utf-8
# Using pretrained vgg for classification
#
import sys, os, glob
import numpy as np
import tensorflow as tf
from tuned_vgg16 import vgg
import matplotlib.pyplot as plt
#import cv2
#
slim = tf.contrib.slim

# Loading tfrecords dataset
class data_generator():
    def __init__(self, FLAGS, name):
        self.steps = FLAGS.num_steps
        if name == 'train':
            self.name = 'training'
            self.b_size = FLAGS.train_batch
            self.tf_dir = FLAGS.training_path

        elif name == 'test':
            self.name = 'testing'
            self.b_size = FLAGS.test_batch
            self.tf_dir = FLAGS.testing_path

        elif name == 'validate':
            self.name = 'validate'
            self.b_size = FLAGS.valid_batch
            self.tf_dir = FLAGS.validate_path
        else:
            print('invalid name, name should be among; train, test, validate')

    def data_list_generator(self):
        # creating data  list here
        record_iterator = tf.python_io.tf_record_iterator(self.tf_dir)
        data_list = []
        # list if (image, label) tuples
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            height = int(example.features.feature[self.name + '/' +'height'].int64_list.value[0])
            width = int(example.features.feature[self.name + '/' + 'width'].int64_list.value[0])
            img_string = (example.features.feature[self.name + '/' + 'image'].bytes_list.value[0])
            lbl_string = (example.features.feature[self.name + '/' + 'label'].bytes_list.value[0])
            img_1d = np.fromstring(img_string, dtype=np.float32)
            lbl_1d = np.fromstring(lbl_string, dtype=np.float32)
            reconstructed_img = img_1d.reshape((height, width, -1))
            data_list.append((reconstructed_img, lbl_1d))
        return data_list


    def numpy_batch_generator(self):
        #
        data_list = self.data_list_generator()
        for j in range(self.steps):
            indexes = np.random.randint(0,len(data_list), self.b_size)
            data_img = np.array([data_list[indexes[i]][0] for i in range(len(indexes))])
            data_lbl = np.array([data_list[indexes[i]][1] for i in range(len(indexes))])
            yield data_img, data_lbl

class tuned_vgg16():
    def __init__(self, FLAGS):
        self.flag = FLAGS
    # preparing training data
    # with graph.as_default():
    def train_and_validate_vgg(self):
        graph = tf.Graph()
        with graph.as_default():
        #
            trn_images = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224,3), name='trn_image')
            trn_labels = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='trn_label')
            #val_x = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224,3), name='val_image')
            val_labels = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='val_label')

            # Load the pretrained VGG16 model from slim extract the fully connected layer
            # before the final output layer

            with slim.arg_scope(vgg.vgg_arg_scope()):
                logits, end_points = vgg.vgg_16(trn_images, num_classes=1000, is_training=False) # trn_images
                fc_6 = end_points['vgg_16/fc6']
                fc_7 = end_points['vgg_16/fc7']

            # Define the only set of weights that we will learn W1 and b1

            W1 =tf.Variable(tf.random_normal([4096,1], mean=0.0, stddev=0.02), name='W1')
            b1 = tf.Variable(tf.random_normal([1], mean=0.0, stddev=0.02), name='b1')
            # -training two layers of the prertained vgg
            W2 =tf.Variable(tf.random_normal([4096,self.flag.num_classes], mean=0.0, stddev=0.02), name='W2')
            b2 = tf.Variable(tf.random_normal([self.flag.num_classes], mean=0.0, stddev=0.02), name='b2')

            #  Reshape the fully connected layer fc_7 and define
            # the logits and probability
            fc_6 = tf.reshape(fc_6, [-1,W1.get_shape().as_list()[0]])
            fc_6 = tf.nn.bias_add(tf.matmul(fc_6,W1),b1)
            #
            fc_7 = tf.reshape(fc_7, [-1, W2.get_shape().as_list()[0]])
            logitx = tf.nn.bias_add(tf.matmul(fc_7,W2),b2)
            softmax_out = tf.nn.softmax(logitx,  name='smax_out')
        #
            # Define Cost and Optimizer
            # Only we wish to learn the weights Wn and b and hence included them in var_list
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=softmax_out, labels=trn_labels))
            # regularization ---
            regularizers = (tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) +
                            tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2) )
            loss += 1e-5 * regularizers
            #
            optimizer = tf.train.AdamOptimizer(learning_rate=self.flag.learning_rate).\
                         minimize(loss, var_list=[W1, b1, W2, b2])
            # measuring the accuracy
            with tf.variable_scope('accuracy') as scope:
                predictions = tf.argmax(tf.nn.softmax(softmax_out), 1, name='final_port')
                true_label = tf.argmax(val_labels, 1)
                equality = tf.equal(predictions, true_label)
                # my accuracy
                accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
                # accuracy with tf metrics
                tfmetrics = tf.metrics.accuracy(true_label, predictions, name=scope)[1]
            #
            # defining saver to keep checkpoints
            saver = tf.train.Saver(max_to_keep=3,  keep_checkpoint_every_n_hours=2)
            # checking for existing meta files
            pre_chkpnt = tf.train.latest_checkpoint(self.flag.chkpnt_path)

            with tf.Session(graph=graph) as sess:
                # writing for tensorboard
                tn_board_writer = tf.summary.FileWriter(self.flag.log_path, sess.graph)
                # preparing summaries
                with tf.name_scope("summaries"):
                    tf.summary.scalar("loss", loss)
                    tf.summary.histogram("histogram_loss",loss)
                    tf.summary.scalar("accuracy", accuracy)
                    tf.summary.scalar("tf_accuracy", tfmetrics)
                merge_all = tf.summary.merge_all()
                #
                init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
                sess.run(init_op)
                #
                # Loading the pre-trained weights for VGG16
                initialize_weights = slim.assign_from_checkpoint_fn(
                                        os.path.join(self.flag.vgg_ckpt_path),
                                        slim.get_model_variables('vgg_16'))
                initialize_weights(sess)
                try:
                    meta_file = pre_chkpnt.split('/')[-1]
                    restorer = tf.train.import_meta_graph(self.flag.chkpnt_path + '/' + meta_file + '.meta')
                    restorer.restore(sess, pre_chkpnt)
                    print('saved model is loaded to continue training ...')

                except:
                    print('meta file was not found, training from pre-trained vgg ...')

                for epoch in range(self.flag.epochs):
                    # Data generator
                    data_gen_train = data_generator(self.flag, 'train')
                    data_gen_val   = data_generator(self.flag, 'validate')
                    try:
                        for step in range(self.flag.num_steps):
                            trn_x, trn_y = next(data_gen_train.numpy_batch_generator())
                            val_x, val_y = next(data_gen_val.numpy_batch_generator())
                            # sess.run(valid_iterator.initializer)
                            val_feed = {trn_images:trn_x, trn_labels: trn_y}
                            #
                            cost_train, _= sess.run([loss, optimizer], feed_dict=val_feed)
                            #measuring the accuracy (using validation data)
                            if step % self.flag.showing_step == 0:
                                val_feed = {trn_images: val_x, val_labels: val_y, trn_labels: trn_y}
                                result_summery, my_acc, tf_acc = sess.run([merge_all, accuracy,  tfmetrics], feed_dict=val_feed)
                                # saving checkpoints here
                                saver.save(sess, self.flag.chkpnt_path + '/vggmt', write_meta_graph=True,
                                           write_state=True, meta_graph_suffix= 'meta',  global_step=step + self.flag.num_steps * epoch)
                                #
                                print('epoch {}, step {}, train loss: {}'.format(epoch, step, cost_train))
                                print('-------- validation accuracy: {}, tf accuracy metic {}'.format(my_acc, tf_acc))
                            # val_tensor_x, val_tensor_y = train_iterator.get_next()
                            tn_board_writer.add_summary(result_summery, step + self.flag.num_steps * epoch)
                    except:
                        pass
                        # trn_images, trn_labels = train_iterator.get_next()
                # saving at the end of epochs
                saver.save(sess, self.flag.chkpnt_path + '/vggmt', write_meta_graph=True, write_state=True,
                           meta_graph_suffix= 'meta',  global_step=step + self.flag.num_steps * epoch)

        os.system('tensorboard --logdir='+self.flag.log_path)

        # representation of test results

    def representer(self, img, treu_label, pred_label):
            #
            true_name = 'sushi' if int(np.argmax(treu_label))==0 else 'sandwich'
            # feeding data to trained network
            l_name = 'sushi'  if np.argmax(pred_label[0])==0 else  'sandwich' # I just edited this part (silly mistake!) in June 26
            print('Predicted probability {}, Predicted label {},True name: {}'
                  .format(np.max(pred_label[0][0]), l_name, true_name))
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.title('Classified Image')
            color = 'g' if l_name==true_name else 'red'
            plt.text(-0, -10, r'Predic: ' + l_name, fontsize=10, color = color)
            plt.text(170, -10, r'True: ' + true_name, fontsize=10, color = 'g')
            plt.imshow(np.squeeze(img, axis=0))
            plt.show()

    def test_vgg(self):
            prediction_graph = tf.Graph()
            with prediction_graph.as_default():
                # loading network
                pre_chkpnt = tf.train.latest_checkpoint(self.flag.chkpnt_path)
                #
                with tf.Session(graph=prediction_graph) as sess:
                    #
                    meta_file = pre_chkpnt.split('/')[-1]
                    restorer = tf.train.import_meta_graph(self.flag.chkpnt_path + '/' + meta_file + '.meta')
                    restorer.restore(sess, pre_chkpnt)
                    img_in = prediction_graph.get_tensor_by_name('trn_image:0')
                    y_smax =  prediction_graph.get_tensor_by_name('smax_out:0')
                    y_out =  prediction_graph.get_tensor_by_name('accuracy/final_port:0') # final_port
                    uninitialized_vars = []
                    for var in tf.all_variables():
                        try:
                            sess.run(var)
                        except tf.errors.FailedPreconditionError:
                            uninitialized_vars.append(var)

                    init_new_vars_op = tf.initialize_variables(uninitialized_vars)
                    sess.run(init_new_vars_op)

                    for test in range(self.flag.num_tests):
                        data_gen = data_generator(self.flag, 'test') # second option is to select dataset
                        img, label = next(data_gen.numpy_batch_generator())

                        output, out_smax = sess.run([y_out, y_smax], feed_dict={img_in : img })
                        self.representer(img, label, out_smax)









