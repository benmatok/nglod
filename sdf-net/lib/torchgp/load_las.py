import numpy as np
import torch
import laspy
import sys
import os
  
def load_las(
    fname : str):
    
    assert fname is not None and os.path.exists(fname), \
        'Invalid file path and/or format, must be an existing Wavefront .obj'

    fname = convert_laz_to_las(fname)

    las = laspy.read(fname)
    coords = np.vstack((las.x, las.y, las.z)).transpose()

    vertices = torch.as_tensor(coords)
    faces = []
    return vertices, faces

def convert_laz_to_las(laz_fname):  
    out_las = laz_fname
    if laz_fname.endswith('.laz'):	                
        out_las = laz_fname.replace('laz', 'las') 
        print('working on file: ', out_las)
        las = laspy.read(laz_fname) # Require numpy version 1.16.6 or earlier to load laz file.
        las = laspy.convert(las)
        las.write(out_las)
        print("Finished converting laz to las without errors")
    
    return out_las