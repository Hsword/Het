import athena
import sys, os
sys.path.append("../../build/lib")

import hetu as ht
import numpy as np

node_id = 0
limit = 10
length = 100
width = 128

array = np.random.random([length, width]).astype(np.float32)

def init_cache():
    cache = ht.LRUCache(limit, length, width, node_id)
    for i in range(limit):
        a = ht.Embedding(i, 0, array[i])
        cache.insert(a)
    return cache

def test_basic_lookup():
    cache = init_cache()
    key = cache.keys()
    value = np.empty((key.size, width), np.float32)
    cache.embedding_lookup(key, value)
    assert(np.all(value == array[key]))

def test_basic_update():
    cache = init_cache()
    key = cache.keys()
    grad = np.random.random([limit, width]).astype(np.float32)
    cache.embedding_update(key, grad)

    value = np.empty((key.size, width), np.float32)
    cache.embedding_lookup(key, value)

    assert(np.mean(value - grad - array[key]) < 1e-6)

def test_miss_lookup():
    cache = init_cache()
    key = cache.keys() + limit
    value = np.empty((key.size, width), np.float32)
    cache.embedding_lookup(key, value)

    assert(np.all(value == 0))

def test_dup_lookup():
    cache = init_cache()
    key = cache.keys()
    key = np.repeat(key, 5)
    value = np.empty((key.size, width), np.float32)
    cache.embedding_lookup(key, value)
    assert(np.all(value == array[key]))

if __name__ == "__main__":
    test_basic_lookup()
    test_basic_update()
    test_miss_lookup()
    test_dup_lookup()
