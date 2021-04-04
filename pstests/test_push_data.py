
from athena import ndarray
from athena import gpu_links as gpu_op
from athena import gpu_ops as ad
import numpy as np

import argparse
import six.moves.cPickle as pickle
import gzip
import sys
import json
import os
import ctypes

def pointer(arr):
    assert(arr.data.c_contiguous)
    assert(arr.dtype == np.long)
    return ctypes.cast(arr.ctypes.data, ctypes.POINTER(ctypes.c_long))

def test():
   ctx = ndarray.cpu(0)
   rank = int(os.environ["WORKER_ID"])
   nrank = int(os.environ["DMLC_NUM_WORKER"])
   arr = ndarray.array(np.random.rand(2,rank+100),ctx = ctx)
   print(arr.asnumpy())

   push_indices = np.array([2*rank+1,2*rank+2])

   if rank == 0:
       pull_indices = np.array([3])
   elif rank == 1:
       pull_indices = np.array([1])

   push_length = np.array([rank+100,rank+100])


   if rank == 0:
     pull_length = np.array([101])
     out_arr = ndarray.array(np.zeros(101),ctx = ctx)
   elif rank == 1:
     pull_length = np.array([100])
     out_arr = ndarray.array(np.zeros(100),ctx = ctx)

   print(out_arr.asnumpy())

   worker_communicate = ad.get_worker_communicate()
   worker_communicate.Push_Data(pointer(push_indices), 2, arr.handle, pointer(push_length))

   worker_communicate.Wait_Push_Data(pointer(push_indices),2);

   worker_communicate.Barrier_Worker()
   worker_communicate.Pull_Data(pointer(pull_indices), 1, out_arr.handle, pointer(pull_length))
   worker_communicate.Wait_Pull_Data(pointer(pull_indices),1);

   print(out_arr.asnumpy())

if __name__ == "__main__":
   file_path = sys.argv[1]
   settings_file = open(file_path)
   settings = json.load(settings_file)

   for key in settings:
     if type(settings[key]) == str:
       os.environ[key] = settings[key]
     else:
       os.environ[key] = str(settings[key]) ## type is str
   if os.environ['DMLC_ROLE'] == "server":
      ad.server_init()
      #ad.StartServer()
      ad.server_finish()
   elif os.environ['DMLC_ROLE'] == "worker": 
      ad.worker_init()
      test()
      ad.worker_finish()
   else:
      ad.scheduler_init()
      ad.scheduler_finish()
    
