# Code source: original BrainGrowth https://github.com/rousseau/BrainGrowth/blob/master/normalisation.py

import numpy as np
from numba import njit, objmode

# Normalize initial mesh coordinates
@njit
def normalise_coord(coordinates0):

  maxx = maxy = maxz = -1e9
  minx = miny = minz = 1e9

  maxx = max(coordinates0[:,0])
  minx = min(coordinates0[:,0])
  maxy = max(coordinates0[:,1])
  miny = min(coordinates0[:,1])
  maxz = max(coordinates0[:,2])
  minz = min(coordinates0[:,2])

  """ center_of_gravity = np.sum(coordinates0, axis=0)
  center_of_gravity /= n_nodes # The center coordinate(x,y,z) """

  # other definition of initial center of gravity COG0
  center_of_gravity_X_0 = 0.5 * (minx + maxx)
  center_of_gravity_Y_0 = 0.5 * (miny + maxy)
  center_of_gravity_Z_0 = 0.5 * (minz + maxz)    
  center_of_gravity = np.array([center_of_gravity_X_0, center_of_gravity_Y_0, center_of_gravity_Z_0])
  with objmode(): 
    print('mesh0 COG = [xG_0:{}, yG_0:{}, zG_0:{}]'.format(center_of_gravity_X_0, center_of_gravity_Y_0, center_of_gravity_Z_0))

  with objmode(): 
    print('minx_0 is {}, maxx_0 is {}'.format(minx, maxx))
    print('miny_0 is {}, maxy_0 is {}'.format(miny, maxy))
    print('minz_0 is {}, maxz_0 is {}'.format(minz, maxz))

  """ if halforwholebrain == "half":
    maxd = max(max(max(abs(maxx-center_of_gravity[0]), abs(minx-center_of_gravity[0])), abs(maxy-miny)), max(abs(maxz-center_of_gravity[2]), abs(minz-center_of_gravity[2])))
    coordinates0[:,0] = -(coordinates0[:,0] - center_of_gravity[0])/maxd
    coordinates0[:,1] = (coordinates0[:,1] - miny)/maxd
    coordinates0[:,2] = -(coordinates0[:,2] - center_of_gravity[2])/maxd """

  maxd = max(max(max(abs(maxx-center_of_gravity[0]), abs(minx-center_of_gravity[0])), max(abs(maxy-center_of_gravity[1]), abs(miny-center_of_gravity[1]))), max(abs(maxz-center_of_gravity[2]), abs(minz-center_of_gravity[2])))
  # new coordinates in barcyenter referential and normalized compared to half maximum coordinates distance to barycenter. 
  coordinates0[:,0] = -(coordinates0[:,0] - center_of_gravity[0])/maxd 
  coordinates0[:,1] = (coordinates0[:,1] - center_of_gravity[1])/maxd
  coordinates0[:,2] = -(coordinates0[:,2] - center_of_gravity[2])/maxd

  with objmode(): 
    print('normalized minx is {}, normalized maxx is {}'.format(min(coordinates0[:,0]), max(coordinates0[:,0])))
    print('normalized miny is {}, normalized maxy is {}'.format(min(coordinates0[:,1]), max(coordinates0[:,1])))
    print('normalized minz is {}, normalized maxz is {}'.format(min(coordinates0[:,2]), max(coordinates0[:,2])))

  coordinates = coordinates0.copy()

  return coordinates