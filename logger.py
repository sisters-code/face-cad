# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy.misc

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x

class Logger(object):
    def __init__(self, log_dir, opt):
        """Create a summary writer logging to log_dir."""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.f = open(os.path.join(log_dir, 'log.txt'), 'w')
        self.tensorboard = SummaryWriter(os.path.join('./log/', opt.arch, opt.expID))

    def write(self, txt):
        self.f.write(txt)

    def close(self):
        self.f.close()

