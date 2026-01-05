import torch

device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs): # 把numpy数组转换为torch张量，并移动到指定设备上
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor): # 把torch张量转换为numpy数组，移动到CPU上并分离计算图
    return tensor.to('cpu').detach().numpy()
