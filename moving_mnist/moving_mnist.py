import os
import gzip
import random
import numpy as np
import torch
import torch.utils.data as data

'''
Download Moving MNIST:
wget http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
'''

def load_mnist(root):
    '''
    Load MNIST dataset
    :param root: the path of MNIST dataset(have train-images-idx3-ubyte.gz file)
    :return: mnist dataset array
    '''
    path = os.path.join(root, 'train-images-idx3-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        '''
        Interpret a buffer as a 1-dimensional array. 
            offset : Start reading the buffer from this offset bytes
        '''
        mnist = mnist.reshape(-1, 28, 28)
        '''
        MNIST shape is (60000, 28, 28)
        '''
    return mnist

def load_fixed_set(root):
    filename = 'mnist_test_seq.npy'
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    '''
        if a array shape is 4
        [np.newaxis, :] shape is (1,4)
        [:, np.newaxis] shape is (4,1)
        e.g. x= np.array([0, 1, 2, 3])
            x[np.newaxis, :] --> array([[0, 1, 2, 3]])
            x[:, np.newaxis] --> array([[0],
                                        [1],
                                        [2],
                                        [3]])
        Moving MNIST shape is (20, 10000, 64, 64, 1)
    '''
    return dataset

class MovingMNIST(data.Dataset):
    def __init__(self, root, is_train=True, n_frames_input=10, n_frames_output=10, num_objects=[2], transform=None):
        super(MovingMNIST, self).__init__()

        self.dataset = None
        if is_train:
            self.mnist = load_mnist(root)
        else:
            if num_objects[0] != 2:
                self.mnist = load_mnist(root)
            else:
                self.dataset = load_fixed_set(root)
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]
        '''
            If is_train is True, need generate Moving MNIST to fed model
            If is_train is False and is not 2 digit in a image, also need generate
            If is_train is False and is 2 digit in a image, use dataset already in path
             
            If generate dataset, length is 10000.
        '''

        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform

        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1

        self.mean = 0
        self.std = 1

    def get_random_trajectory(self, seq_length):
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        #return a [0.0, 1.0] float number
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            #Take a step along velocity
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            #Bounce off edges
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_x[i] = x
            start_y[i] = y

        # Scale to the size of the canvas
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        data = np.zeros((self.n_frames_total, self.image_size_, self.image_size_), dtype = np.float32)
        for n in range(num_digits):
            start_y, start_x = self.get_random_trajectory(self.n_frames_total)
            ind = random.randint(0, self.mnist.shape[0] - 1)#random a MNIST sample
            digit_image = self.mnist[ind]
            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                #make whole MNIST digit in a (64, 64) image

                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis] #data shape is (20, 64, 64, 1)
        return data

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        if self.is_train or self.num_objects[0] != 2:
            num_digits = random.choice(self.num_objects)
            '''
            lst = [a, b, c] 
            random.choice(lst) --> a/b/c
            '''
            images = self.generate_moving_mnist(num_digits=num_digits)
        else:
            images = self.dataset[:, idx, ...] #(20, 10000, 64, 64, 1)

        r = 1 #like patch size
        w = int(64 / r)
        images = images.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r*r, w, w))
        #length, channel, height, width

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        output = torch.from_numpy(output / 255.0).contiguous().float()
        input = torch.from_numpy(input / 255.0).contiguous().float()
        return input, output

    def __len__(self):
        return self.length

def load_data(batch_size, data_root, num_workers):
    train_set = MovingMNIST(root=data_root, is_train=True, n_frames_input=10, n_frames_output=10, num_objects=[3])
    test_set = MovingMNIST(root=data_root, is_train=False, n_frames_input=10, n_frames_output=10, num_objects=[3])

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True)
    dataloader_validation = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=True)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=True)

    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, std



