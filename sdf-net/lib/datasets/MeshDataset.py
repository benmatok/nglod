# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os

import torch
from torch.utils.data import Dataset

import numpy as np
import mesh2sdf
import laspy
from scipy import spatial

from lib.torchgp import load_obj, point_sample, sample_surface, compute_sdf, normalize, load_las
from lib.PsDebugger import PsDebugger

from lib.utils import PerfTimer, setparam

import collections
import operator
from datetime import datetime

BT = collections.namedtuple("BT", ["value", "left", "right"])
BT.__doc__ = """
A Binary Tree (BT) with a node value, and left- and
right-subtrees.
"""

def kdtree(points):
    """Construct a k-d tree from an iterable of points.
    
    This algorithm is taken from Wikipedia. For more details,
    
    > https://en.wikipedia.org/wiki/K-d_tree#Construction
    
    """
    k = len(points[0])
    
    def build(*, points, depth):
        """Build a k-d tree from a set of points at a given
        depth.
        """
        if len(points) == 0:
            return None
        
        points.sort(key=operator.itemgetter(depth % k))
        middle = len(points) // 2
        
        return BT(
            value = points[middle],
            left = build(
                points=points[:middle],
                depth=depth+1,
            ),
            right = build(
                points=points[middle+1:],
                depth=depth+1,
            ),
        )
    
    return build(points=list(points), depth=0)

NNRecord = collections.namedtuple("NNRecord", ["point", "distance"])
NNRecord.__doc__ = """
Used to keep track of the current best guess during a nearest
neighbor search.
"""

def find_nearest_neighbor(*, tree, point):
    """Find the nearest neighbor in a k-d tree for a given
    point.
    """
    k = len(point)
    
    best = None
    def search(*, tree, depth):
        """Recursively search through the k-d tree to find the
        nearest neighbor.
        """
        nonlocal best
        
        if tree is None:
            return
        
        distance = SED(tree.value, point)
        if best is None or distance < best.distance:
            best = NNRecord(point=tree.value, distance=distance)
        
        axis = depth % k
        diff = point[axis] - tree.value[axis]
        if diff <= 0:
            close, away = tree.left, tree.right
        else:
            close, away = tree.right, tree.left
        
        search(tree=close, depth=depth+1)
        if diff**2 < best.distance:
            search(tree=away, depth=depth+1)
    
    search(tree=tree, depth=0)
    return best.point

def nearest_neighbor_kdtree(*, query_points, reference_points):
    """Use a k-d tree to solve the "Nearest Neighbor Problem"."""
    tree = kdtree(reference_points)
    print("Finished creating kdtree: ", datetime.now())
    return {
        query_p: find_nearest_neighbor(tree=tree, point=query_p)
        for query_p in query_points
    }

def SED(X, Y):
    """Compute the squared Euclidean distance between X and Y."""
    return np.sqrt(sum((i-j)**2 for i, j in zip(X, Y)))

def nearest_neighbor_bf(*, query_points, reference_points):
    """Use a brute force algorithm to solve the
    "Nearest Neighbor Problem".
    """
    return [
        SED(query_p, min(
            reference_points,
            key=lambda X: SED(X, query_p),
        ))
        for query_p in query_points
    ]

class MeshDataset(Dataset):
    """Base class for single mesh datasets."""

    def __init__(self, 
        args=None, 
        dataset_path = None,
        raw_obj_path = None,
        sample_mode = None,
        get_normals = None,
        seed = None,
        num_samples = None,
        trim = None,
        sample_tex = None
    ):
        self.args = args
        self.dataset_path = setparam(args, dataset_path, 'dataset_path')
        self.raw_obj_path = setparam(args, raw_obj_path, 'raw_obj_path')
        self.sample_mode = setparam(args, sample_mode, 'sample_mode')
        self.get_normals = setparam(args, get_normals, 'get_normals')
        self.num_samples = setparam(args, num_samples, 'num_samples')
        self.trim = setparam(args, trim, 'trim')
        self.sample_tex = setparam(args, sample_tex, 'sample_tex')

        # Possibly remove... or fix trim obj
        #if self.raw_obj_path is not None and not os.path.exists(self.dataset_path):
        #    _, _, self.mesh = trim_obj_to_file(self.raw_obj_path, self.dataset_path)
        #elif not os.path.exists(self.dataset_path):
        #    assert False and "Data does not exist and raw obj file not specified"
        #else:
        
        if self.sample_tex:
            out = load_obj(self.dataset_path, load_materials=True)
            self.V, self.F, self.texv, self.texf, self.mats = out
        else:
            self.V, self.F = load_obj(self.dataset_path) # load_obj / las
        self.V = normalize(self.V)
        #self.mesh = self.V[self.F]
        self.resample()

    def resample(self):
        """Resample SDF samples."""

        self.nrm = None
        if self.get_normals:
            self.pts, self.nrm = sample_surface(self.V, self.F, self.num_samples*5)
            self.nrm = self.nrm.cpu()
        else:
            self.pts = point_sample(self.V, self.F, self.sample_mode, self.num_samples)

        #self.d = compute_sdf(self.V.cuda(), self.F.cuda(), self.pts.cuda())   
        
        #reference_points = [ (1, 2), (3, 2), (4, 1), (3, 5) ]
        #query_points = [
        #    (3, 4), (5, 1), (7, 3), (8, 9), (10, 1), (3, 3)
        #]

        print("Started: ", datetime.now())
        #nearest_neighbor_bf(
        #    reference_points = self.V,
        #    query_points = self.pts,
        #)

        #tree = kdtree(self.V)
        #find_nearest_neighbor(tree=tree, point=(10, 1))
        self.d = nearest_neighbor_kdtree(
            reference_points = self.V,
            query_points = self.pts,
        )
        print("Finished: ", datetime.now())

        #tree = spatial.KDTree(self.V)
        #self.d = [list[1] for list in tree.query(self.V, k = 2)[0]]
        self.d = torch.as_tensor(self.d)
        print("self d: ", self.d.size())

        self.d = self.d[...,None]
        self.d = self.d.cpu()
        self.pts = self.pts.cpu()

    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        if self.get_normals:
            return self.pts[idx], self.d[idx], self.nrm[idx]
        elif self.sample_tex:
            return self.pts[idx], self.d[idx], self.rgb[idx]
        else:
            return self.pts[idx], self.d[idx]
            
    def __len__(self):
        """Return length of dataset (number of _samples_)."""

        return self.pts.size()[0]

    def num_shapes(self):
        """Return length of dataset (number of _mesh models_)."""

        return 1
