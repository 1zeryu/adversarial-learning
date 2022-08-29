import torch
import numpy as np

class Blackpepper(object):
    """_summary_
    Args:
        attack (_type_): adding black pepper noise to images
    """
    def __init__(self, nums):
        """_summary_

        Args:
            nums (_type_): the number of black pepper pixels to adding
        """
        self.nums = nums
        self.pixels = {0: (0, 0, 0),
                       1: (255, 255, 255)}
    
    def __call__(self, inputs):
        
        images = inputs.clone().detach()
        
        adv_images = images.clone().detach()
        print(adv_images.shape)
        c, h, w = inputs.shape[:3]
        rows = np.random.randint(0, h, (self.nums), dtype=int)
        cols = np.random.randint(0, w, (self.nums), dtype=int)
        for i in range(self.nums):
            if i % 2 == 1:
                adv_images[:,c,rows[i], cols[i]] = torch.tensor(self.pixels[1])
            else:
                adv_images[:,c,rows[i], cols[i]] = torch.tensor(self.pixels[0])
                
        return adv_images
    
