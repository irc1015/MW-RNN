from moving_mnist import load_data as load_mmnist

def load_data(dataname, batch_size, data_root, num_workers, **kwargs):
    if dataname =='mmnist':
        return load_mmnist(batch_size=batch_size,
                           data_root=data_root,
                           num_workers=num_workers)