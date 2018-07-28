#
# getting raw images and making tfrecords
#
import tensorflow as tf
import numpy as np
import cv2
import os
#
# setting parameters by defining parser here
#
class image_write_read_tf():
    def __init__(self):
        address = os.getcwd()
        self.size     = (224, 224, 3)
        self.height   = self.size[0]
        self.width    = self.size[1]
        self.train_ratio = .8
        self.validate_ratio = .1
        self.test_ration = .1
        # i changed this address in june 26
        self.susand_train_path    = address + '/sushi_or_sandwich_photos/'
        self.train_filename_tf    = address + '/training_susand_01.tfrecords'
        self.validate_filename_tf = address + '/validate_susand_01.tfrecords'
        self.test_filename_tf     = address + '/testing_susand_01.tfrecords'

    # encoding
    def inp2int(self, input):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[input]))
    def inp2byte(self, input):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value = [input]))

    def train_valid_test(self):
        # path
        for _, folders, _ in os.walk(self.susand_train_path):
            label = folders
            break
        addrs = {}
        for folder in label:
            for _, _, files in os.walk(self.susand_train_path + folder):
                for file in files:
                    addrs.update({self.susand_train_path + folder + '/' + file: folder})
        total_data_num = len(addrs)
        # shuffling data list (very important)
        temp = list(addrs.items())
        np.random.shuffle(temp)
        np.random.shuffle(temp)
        np.random.shuffle(temp)
        addrs = dict(temp)
        # making train, validation and test data by keeping the image addreses
        train_addrs   = list(addrs.keys())[0:int(self.train_ratio * total_data_num)]
        train_labels  = list(addrs.values())[0:int(self.train_ratio * total_data_num)]
        val_addrs     = list(addrs.keys())[int(self.train_ratio * total_data_num):int((self.train_ratio + self.validate_ratio) * total_data_num)]
        val_labels    = list(addrs.values())[int(self.train_ratio * total_data_num):int((self.train_ratio + self.validate_ratio) * total_data_num)]
        test_addrs    = list(addrs.keys())[int((self.train_ratio + self.validate_ratio) * total_data_num):]
        test_labels   = list(addrs.values())[int((self.train_ratio + self.validate_ratio) * total_data_num):]
        return train_addrs, train_labels, val_addrs, val_labels, test_addrs, test_labels

    def tf_writer(self, data_adr, label_nm, data_dir):
        writer = tf.python_io.TFRecordWriter(data_dir)
        for i in range(len(data_adr)):
            # print how many images are saved every 1000 images
            if not i % 10:
                print('progress: {} %'.format(int((i/len(data_adr))*100)+1))
                #sys.stdout.flush()
            # Load to byte processes
            label = np.zeros((1,2),dtype=np.float32)[0]
            if label_nm[i] == 'sushi':
                label[0] = 1
            else:
                label[1] = 1
            label = label.tostring()
            label = tf.compat.as_bytes(label)
            # image to byte processes
            img = cv2.cvtColor(cv2.imread(data_adr[i]), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(self.height, self.width))
            # standardization
            mean = 10
            img = (img-mean)/255
            img = img.astype(np.float32) # I am using this for special type of dataset creation
            # Create a feature
            img = img.tostring()
            img = tf.compat.as_bytes(img)
            # this part can be done in an advanced way to improve the coding
            root_name = data_dir.split('/')[-1].split('_')[0]
            feature = {root_name + '/label'   : self.inp2byte(label),
                       root_name + '/height'  : self.inp2int(self.height),
                       root_name + '/width'   : self.inp2int(self.width),
                       root_name + '/filename': self.inp2byte(bytes(label_nm[i], 'utf-8')),
                       root_name + '/image'   : self.inp2byte(img)
                       }
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # current_image_object.image # cropped image with size 299x299
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        writer.close()
        #sys.stdout.flush()

    def cov2tfrecord(self):
        train_d, train_lab, val_d, val_lab, test_d, test_lab = self.train_valid_test()
        # writing train, validation and test data in separate files
        # making train data
        print('creating train data ...')
        if not os._exists(self.train_filename_tf):
            open(self.train_filename_tf,'a').close()
        self.tf_writer(train_d, train_lab, self.train_filename_tf)
        print('train data is created.')

        print('creating validation data ...')
        if not os._exists(self.validate_filename_tf):
            open(self.validate_filename_tf,'a').close()
        self.tf_writer(val_d, val_lab, self.validate_filename_tf)
        print('validation data is created.')

        print('creating test data ...')
        if not os._exists(self.test_filename_tf):
            open(self.test_filename_tf,'a').close()
        self.tf_writer(test_d, test_lab, self.test_filename_tf)
        print('test data is created.')
        print('Creating tfrecords data is finished.')

if __name__ == '__main__':
    creat_data = image_write_read_tf()
    creat_data.cov2tfrecord()