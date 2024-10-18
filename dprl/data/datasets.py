from collections import defaultdict
from typing import Callable
import h5py
import torch
from torch.utils.data import Dataset
from os import PathLike
import os
import numpy as np

"""
This class is a hot mess
You can use inheritance or fp to reduce code by like 80%
but this is like the last priority
"""

class LiberoDatasetAdapter(Dataset):
    def __init__(self, root_dir : str,
                 slice_len : int = 50,
                 transform = None,
                 frameskip : int = 1):
        assert os.path.isdir(root_dir), "Root directory must exist"
        self.idx_to_slice = {}
        self.root_dir = root_dir
        self.slice_len = slice_len
        self.transform = transform
        self.frameskip = frameskip
        for scene in os.listdir(root_dir):
            if scene.split('.')[-1] != "hdf5":
                continue
            with h5py.File(os.path.join(root_dir, scene), 'r') as f:
                for episode in f['data'].keys():
                    ep_len = len(f['data'][episode]['actions'])
                    
                    if ep_len < slice_len:
                        continue
                    slice_start = 0
                    while True:
                        if slice_start + slice_len >= ep_len:
                            self.idx_to_slice[len(self.idx_to_slice)] = (scene, episode, ep_len - slice_len)
                            break
                        self.idx_to_slice[len(self.idx_to_slice)] = (scene, episode, slice_start)
                        slice_start += slice_len
        self.f = ""
        self.fp = None
        
    def __len__(self):
        return len(self.idx_to_slice)
    
    def __getitem__(self, index):

        scene, episode, slice_start = self.idx_to_slice[index]
        if self.f != scene or self.fp == None:
            if self.fp != None:
                self.fp.close()
                self.fp = None
            self.fp = h5py.File(os.path.join(self.root_dir, scene))
            self.f = scene
        
        ep_gp = self.fp['data'][episode]
                
        data = {
            'observations': ep_gp['obs']['agentview_rgb'][slice_start:slice_start+self.slice_len:self.frameskip],
            'actions': ep_gp['actions'][slice_start:slice_start+self.slice_len:self.frameskip],
            'rewards': ep_gp['rewards'][slice_start:slice_start+self.slice_len:self.frameskip]
        }
        
        if self.transform != None:
            data = self.transform(data)
                
        return data        
        
class MinigridAdapterDataset(Dataset):
    def __init__(self, root_dir : str, slice_len : int = 10, transform = None):
        self.idx_to_slice = {}
        
        