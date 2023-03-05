import numpy as np
from skimage.metrics import structural_similarity

def MAE(pred, true):
    return np.mean(np.abs(pred - true), axis = (0, 1)).sum()

def MSE(pred, true):
    return np.mean((pred - true) ** 2, axis = (0, 1)).sum()

def PSNR(pred, true):
    mse = np.mean((np.uint8(pred * 255) - np.uint8(true * 255)) ** 2)
    return 20 * np.log10(255) - 10 * np.log10(mse)

def metric( pred, true, mean, std, return_ssim_psnr=False, clip_range=[0,1]):
    pred = pred * std + mean
    true = true * std + mean
    mae = MAE(pred, true)
    mse = MSE(pred, true)

    if return_ssim_psnr:
        pred = np.maximum(pred, clip_range[0])
        pred = np.minimum(pred, clip_range[1])
        ssim, psnr = 0, 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                #batch, time, channel, height, width
                ssim += structural_similarity(pred[b, f], true[b, f], channel_axis=0)
                #this parameter indicates which axis of the array corresponds to channels.
                psnr += PSNR(pred[b, f], true[b, f])

        ssim = ssim / (pred.shape[0] * pred.shape[1])
        psnr = psnr / (pred.shape[0] * pred.shape[1])
        return mse, mae, ssim, psnr
    else:
        return mse, mae

