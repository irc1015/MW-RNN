import numpy as np
import math


def reserve_schedule_sampling(in_shape, batch_size, input_length, total_length, r_sampling_step_1,
                              r_sampling_step_2, r_exp_alpha, itr, patch_size):
    _, C, H, W = in_shape

    if itr < r_sampling_step_1:
        r_eta = 0.5
    elif itr < r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - r_sampling_step_1) / r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < r_sampling_step_1:
        eta = 0.5
    elif itr < r_sampling_step_2:
        eta = 0.5 - (0.5 / (r_sampling_step_2 - r_sampling_step_1)) * (itr - r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample((batch_size, input_length - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample((batch_size, total_length - input_length - 1))
    true_token = (random_flip < eta)

    ones = np.ones((C*patch_size*patch_size, H//patch_size, W//patch_size))
    zeros = np.zeros((C*patch_size*patch_size, H//patch_size, W//patch_size))

    real_input_flag = []
    for i in range(batch_size):
        for j in range(total_length - 2):
            if j < input_length - 1:#0-8
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:#9-17
                if true_token[i, j - (input_length - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag, (batch_size, total_length - 2,
                                                   C*patch_size*patch_size, H//patch_size, W//patch_size))
    return real_input_flag