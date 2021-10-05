import torch

import socket
import threading
import os

from lqrlearning.utils import file_utils

def launch_tensorboard(tensorboard_dir, port, erase=True):
    if erase:
        file_utils.create_empty_directory(tensorboard_dir)

    # Use threading so tensorboard is automatically closed on process end
    command = f'tensorboard --bind_all --port {port} '\
              f'--logdir {tensorboard_dir} > /dev/null '\
              f'--window_title {socket.gethostname()} 2>&1'
    t = threading.Thread(target=os.system, args=(command,))
    t.start()

    print(f'Launching tensorboard on http://localhost:{port}')

def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def numpy(tensor):
    return tensor.detach().cpu().numpy()
