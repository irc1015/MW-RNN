import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    #[batch, length, channel, height, width] --> [batch, length, height, width, channel]
    img_tensor_t = np.transpose(img_tensor, (0, 1, 3, 4, 2))

    batch_size = np.shape(img_tensor_t)[0]
    seq_length = np.shape(img_tensor_t)[1]
    img_height = np.shape(img_tensor_t)[2]
    img_width = np.shape(img_tensor_t)[3]
    num_channels = np.shape(img_tensor_t)[4]

    a = np.reshape(img_tensor_t, [batch_size, seq_length,
                                img_height // patch_size, patch_size,
                                img_width // patch_size, patch_size,
                                num_channels])
    b = np.transpose(a, [0, 1, 2, 3, 4, 5, 6])

    patch_tensor = np.reshape(b, [batch_size, seq_length,
                                  img_height // patch_size, img_width // patch_size,
                                  patch_size * patch_size * num_channels])

    patch_tensor_t = np.transpose(patch_tensor, (0, 1, 4, 2, 3))
    return patch_tensor_t

def reshape_patch_back(patch_tensor, patch_size):
    assert 5 == patch_tensor.ndim
    # [batch, length, channel, height, width] --> [batch, length, height, width, channel]
    patch_tensor_t = np.transpose(patch_tensor, (0, 1, 3, 4, 2))
    batch_size = np.shape(patch_tensor_t)[0]
    seq_length = np.shape(patch_tensor_t)[1]
    patch_height = np.shape(patch_tensor_t)[2]
    patch_width = np.shape(patch_tensor_t)[3]
    channels = np.shape(patch_tensor_t)[4]
    img_channels = channels // (patch_size*patch_size)

    a = np.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    b = np.transpose(a, [0,1,2,3,4,5,6])
    img_tensor = np.reshape(b, [batch_size, seq_length,
                                patch_height*patch_size,
                                patch_width*patch_size,
                                img_channels])
    img_tensor_t = np.transpose(img_tensor, (0, 1, 4, 2, 3))
    return img_tensor_t

