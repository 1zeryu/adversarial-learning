import logging
from torch.utils.tensorboard import SummaryWriter
import time
import os
import datetime

class log(object) :
    def __init__(self, filename) :
        if not os.path.isdir('logs') :
            os.mkdir('logs')
        loggerfile = './logs/' + time.strftime("%y%m%d%H%M%S", time.localtime()) + filename + '.log'
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(loggerfile)
        formatter = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    
    
    def info(self, information):
        self.logger.info(information)

    def close(self):
        self.logger.close()

class writer(object):
    def __init__(self, filename):
        if not os.path.isdir('runs') :
            os.mkdir('runs')
        log_dir = './runs/' + time.strftime("%y%m%d%H%M%S", time.localtime()) + filename
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def test_acc(self, acc, epoch):
        self.writer.add_scalar('acc/test', acc, epoch)

    def train_acc(self, acc, epoch):
        self.writer.add_scalar('acc/train', acc, epoch)

    def train_loss(self, loss, epoch):
        self.writer.add_scalar('loss/train', loss, epoch)
    
    def test_loss(self, loss, epoch):
        self.writer.add_scalar('loss/test', loss, epoch)
    
    def images(self, batch_images, epoch, name='my_image_batch'):
        self.writer.add_images(name, batch_images.cpu().numpy(), epoch)
        
    def close(self):
        self.writer.close()

class timer(object):
    def logtime(self):
        return time.strftime('%a %b %d %H:%M:%S %Y', time.localtime())
    
    def filetime(self):
        return time.strftime("%y-%m-%d-%H:%M:%S", time.localtime())

    def get_hms(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)

        return h, m, s

    start_time = 0
    epoch_time = 0
    running_time = 0
    def setStart(self):
        self.start_time = time.time()
    
    def setEpoch(self):
        self.epoch_time = time.time() - self.start_time
        self.running_time += self.epoch_time

    def runtime(self):
        return '| Running time : %d:%02d:%02d'  %(self.get_hms(self.running_time))

