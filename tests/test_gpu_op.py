import numpy as np
from athena import ndarray
from athena import gpu_links as gpu_op
from athena import gpu_ops as ad
# import tensorflow as tf


def test_array_set():
    ctx = ndarray.gpu(0)
    shape = (500, 200)
    # oneslike
    arr_x = ndarray.empty(shape, ctx=ctx)
    gpu_op.array_set(arr_x, 1.)
    x = arr_x.asnumpy()
    np.testing.assert_allclose(np.ones(shape), x)
    # zeroslike
    gpu_op.array_set(arr_x, 0.)
    x = arr_x.asnumpy()
    np.testing.assert_allclose(np.zeros(shape), x)


def test_broadcast_to():
    ctx = ndarray.gpu(0)
    shape = (200, 300)
    to_shape = (130, 200, 300)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(to_shape, ctx=ctx)
    gpu_op.broadcast_to(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np.broadcast_to(x, to_shape), y)


def test_reduce_sum_axis_zero():
    ctx = ndarray.gpu(0)
    shape = (20, 1, 1)
    temp_shape = list(shape)
    temp_shape[0] = (temp_shape[0] + 1) // 2
    temp_shape = tuple(temp_shape) 
    to_shape = (1, 1)
    x = np.random.uniform(0, 20, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(to_shape, ctx=ctx)
    arr_workspace = ndarray.empty(shape = temp_shape, ctx = ctx)
    gpu_op.reduce_sum_axis_zero(arr_x, arr_y, arr_workspace)
    y = arr_y.asnumpy()
    y_ = np.sum(x, axis=0)
    for index, _ in np.ndenumerate(y):
        v = y[index]
        v_ = y_[index]
        if abs((v - v_) / v_) > 1e-4:
            print(index, v, v_)
    np.testing.assert_allclose(np.sum(x, axis=0), y, rtol=1e-5)


def test_matrix_elementwise_add():
    ctx = ndarray.gpu(0)
    shape = (500, 200)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    y = np.random.uniform(0, 10, size=shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_add(arr_x, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(x + y, z, rtol=1e-5)


def test_matrix_elementwise_add_by_const():
    shape = (2000, 3000)
    ctx = ndarray.gpu(0)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    val = np.random.uniform(-5, 5)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_add_by_const(arr_x, val, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(x + val, y, rtol=1e-5)


def test_matrix_elementwise_multiply():
    ctx = ndarray.gpu(0)
    shape = (500, 200)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    y = np.random.uniform(0, 10, size=shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_multiply(arr_x, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(x * y, z, rtol=1e-5)


def test_matrix_elementwise_multiply_by_const():
    shape = (2000, 3000)
    ctx = ndarray.gpu(0)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    val = np.random.uniform(-5, 5)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_multiply_by_const(arr_x, val, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(x * val, y, rtol=1e-5)


def test_matrix_elementwise_divide():
    ctx = ndarray.gpu(0)
    shape = (500, 200)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    y = np.random.uniform(1, 10, size=shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_divide(arr_x, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(x / y, z, rtol=1e-5)


def test_matrix_elementwise_divide_const():
    shape = (2000, 3000)
    ctx = ndarray.gpu(0)
    val = np.random.uniform(-5, 5)
    x = np.random.uniform(1, 10, size=shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_divide_const(val, arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(val / x, y, rtol=1e-5)


def test_matrix_opposite():
    shape = (2000, 2500)
    ctx = ndarray.gpu(0)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    gpu_op.matrix_opposite(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(-x, y)


def test_matrix_multiply():
    ctx = ndarray.gpu(0)
    x = np.random.uniform(0, 10, size=(500, 700)).astype(np.float32)
    y = np.random.uniform(0, 10, size=(700, 1000)).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty((500, 1000), ctx=ctx)
    gpu_op.matrix_multiply(arr_x, False, arr_y, False, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.dot(x, y), z, rtol=1e-5)

    x = np.random.uniform(0, 10, size=(1000, 500)).astype(np.float32)
    y = np.random.uniform(0, 10, size=(2000, 500)).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty((1000, 2000), ctx=ctx)
    gpu_op.matrix_multiply(arr_x, False, arr_y, True, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.dot(x, np.transpose(y)), z, rtol=1e-5)
    
    x = np.random.uniform(0, 10, size=(500, 1000)).astype(np.float32)
    y = np.random.uniform(0, 10, size=(2000, 500)).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty((1000, 2000), ctx=ctx)
    gpu_op.matrix_multiply(arr_x, True, arr_y, True, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.dot(np.transpose(x), np.transpose(y)), z,
                               rtol=1e-5)


def test_matrix_sqrt():
    shape = (2000, 2500)
    ctx = ndarray.gpu(0)
    x = np.random.uniform(0, 10, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    gpu_op.matrix_sqrt(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np.sqrt(x), y, rtol=1e-5)


def test_matrix_rsqrt():
    shape = (2000, 2500)
    ctx = ndarray.gpu(0)
    x = np.random.uniform(0, 10, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    gpu_op.matrix_rsqrt(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(1 / np.sqrt(x), y, rtol=1e-5)


def test_relu():
    shape = (2000, 2500)
    ctx = ndarray.gpu(0)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    gpu_op.relu(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np.maximum(x, 0).astype(np.float32), y)


def test_relu_gradient():
    shape = (2000, 2500)
    ctx = ndarray.gpu(0)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    grad_x = np.random.uniform(-5, 5, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_grad_x = ndarray.array(grad_x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    gpu_op.relu_gradient(arr_x, arr_grad_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(((x > 0) * grad_x).astype(np.float32), y)


def test_softmax():
    def softmax_func(y):
        """Numerically stable softmax."""
        b = y - np.max(y, axis=1, keepdims=True)
        expb = np.exp(b)
        softmax = expb / np.sum(expb, axis=1, keepdims=True)
        return softmax
    ctx = ndarray.gpu(0)
    shape = (400, 1000)
    x = np.random.uniform(-5, 5, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    gpu_op.softmax(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(softmax_func(x), y, rtol=1e-5)


def test_softmax_cross_entropy():
    def softmax_func(y):
        """Numerically stable softmax."""
        b = y - np.max(y, axis=1, keepdims=True)
        expb = np.exp(b)
        softmax = expb / np.sum(expb, axis=1, keepdims=True)
        return softmax
    ctx = ndarray.gpu(0)
    shape = (400, 1000)
    y = np.random.uniform(-5, 5, shape).astype(np.float32)
    y_ = np.random.uniform(-5, 5, shape).astype(np.float32)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_y_ = ndarray.array(y_, ctx=ctx)
    arr_out = ndarray.empty((400,), ctx=ctx)
    gpu_op.softmax_cross_entropy(arr_y, arr_y_, arr_out)
    out = arr_out.asnumpy()
    # numpy calculation
    cross_entropy = -np.sum(y_ * np.log(softmax_func(y)), axis=1)
    np.testing.assert_allclose(cross_entropy, out, rtol=1e-4)
    # test cudnn
    gpu_op.CuDNN_softmax_cross_entropy(arr_y, arr_y_, arr_out)
    out = arr_out.asnumpy()
    np.testing.assert_allclose(cross_entropy, out, rtol=1e-4)


def test_softmax_cross_entropy_gradient():
    def softmax_func(y):
        """Numerically stable softmax."""
        b = y - np.max(y, axis=1, keepdims=True)
        expb = np.exp(b)
        softmax = expb / np.sum(expb, axis=1, keepdims=True)
        return softmax
    ctx = ndarray.gpu(0)
    shape = (400, 1000)
    y = np.random.uniform(-5, 5, shape).astype(np.float32)
    y_ = np.random.uniform(-5, 5, shape).astype(np.float32)
    grad = np.random.uniform(-5, 5, (400,)).astype(np.float32)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_y_ = ndarray.array(y_, ctx=ctx)
    arr_grad = ndarray.array(grad, ctx=ctx)
    arr_out = ndarray.empty(shape, ctx=ctx)
    gpu_op.softmax_cross_entropy_gradient(arr_y, arr_y_, arr_grad, arr_out)
    out = arr_out.asnumpy()
    # numpy calculation
    np_grad = (softmax_func(y) + -1 * y_) * np.expand_dims(grad, -1)
    np.testing.assert_allclose(np_grad, out, rtol=1e-4, atol=1e-8)

    # test cudnn
    gpu_op.CuDNN_softmax_cross_entropy_gradient(arr_grad, arr_y, arr_y_, arr_out)
    out = arr_out.asnumpy()
    np.testing.assert_allclose(np_grad, out, rtol=1e-4, atol=1e-8)


def test_conv2d():
    ctx = ndarray.gpu(0)
    # im2col and np_conv2d are helper functions
    def im2col(X, filter_H, filter_W, padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1

        y_row_size = C * filter_H * filter_W
        y_col_size = out_H * out_W
        y_shape = (N, y_row_size, y_col_size)
        Y = np.empty(y_shape, dtype = X.dtype)

        for batch_index in range(N):
            for col_index in range(y_col_size):
              out_y = col_index // out_W
              out_x = col_index % out_W
              in_y = out_y * stride - padding
              in_x = out_x * stride - padding
              row_idx = 0
              for c in range(0, C):
                  for y in range(in_y, in_y + filter_H):
                      for x in range(in_x, in_x + filter_W):
                          if (x < 0 or x >= W or y < 0 or y >= H):
                              Y[batch_index, row_idx, col_index] = 0
                          else:
                              Y[batch_index, row_idx, col_index] = X[batch_index, c, y, x]
                          row_idx += 1
        return Y

    def np_conv2d(X, Filter, padding=0, stride=1):
        """Implement a conv2d as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1

        im2col_matrix = im2col(X, filter_H, filter_W, padding, stride)
        filter_matrix = Filter.reshape(filter_outChannel, -1)
        print("shape", im2col_matrix.shape)
        print("shape", filter_matrix.shape)
        print("shape", np.matmul(filter_matrix, im2col_matrix).shape)
        return np.matmul(filter_matrix, im2col_matrix).reshape(N, filter_outChannel, out_H, out_W)
        #return im2col_matrix

    shapeX = (100, 3, 28, 28)
    shapeF = (10, 3, 5, 5)
    shapeY = (100, 10, 24, 24)
    shapeW = (100, 3*5*5, 24*24)
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    f = np.random.uniform(0, 10, size=shapeF).astype(np.float32)
    y = np.zeros(shapeY).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_f = ndarray.array(f, ctx=ctx)
    arr_y = ndarray.empty(shapeY, ctx=ctx)
    arr_workspace = ndarray.empty(shapeW, ctx=ctx)

    gpu_op.conv2d(arr_x, arr_f, arr_y, arr_workspace)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np_conv2d(x, f), y, rtol=1e-5)


def test_conv2d_Gradient():
    ctx = ndarray.gpu(0)
    def im2col(X, filter_H, filter_W, padding, stride):
      N, C, H, W = X.shape
      assert (H + 2 * padding - filter_H) % stride == 0
      assert (W + 2 * padding - filter_W) % stride == 0
      out_H = (H + 2 * padding - filter_H) // stride + 1
      out_W = (W + 2 * padding - filter_W) // stride + 1

      y_row_size = C * filter_H * filter_W
      y_col_size = out_H * out_W
      y_shape = (N, y_row_size, y_col_size)
      Y = np.empty(y_shape, dtype = X.dtype)

      for batch_index in range(N):
          for col_index in range(y_col_size):
            out_y = col_index // out_W
            out_x = col_index % out_W
            in_y = out_y * stride - padding
            in_x = out_x * stride - padding
            row_idx = 0
            for c in range(0, C):
                for y in range(in_y, in_y + filter_H):
                    for x in range(in_x, in_x + filter_W):
                        if (x < 0 or x >= W or y < 0 or y >= H):
                            Y[batch_index, row_idx, col_index] = 0
                        else:
                            Y[batch_index, row_idx, col_index] = X[batch_index, c, y, x]
                        row_idx += 1
      return Y
    def np_conv2d(X, Filter, padding=0, stride=1):
        """Implement a conv2d as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1

        im2col_matrix = im2col(X, filter_H, filter_W, padding, stride)
        filter_matrix = Filter.reshape(filter_outChannel, -1)
        # print("shape", im2col_matrix.shape)
        # print("shape", filter_matrix.shape)
        # print("shape", np.matmul(filter_matrix, im2col_matrix).shape)
        return np.matmul(filter_matrix, im2col_matrix).reshape(N, filter_outChannel, out_H, out_W)
        #return im2col_matrix
    def im2col_transpose(X, filter_H, filter_W, Y , padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1
        _, y_row_size, y_col_size = Y.shape

        der_X_shape = (N, C, H, W)
        der_X = np.zeros(der_X_shape, dtype = X.dtype)


        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index // out_W
                out_x = col_index % out_W
                in_y = out_y * stride - padding
                in_x = out_x * stride - padding
                row_idx = 0
                for c in range(0, C):
                    for y in range(in_y, in_y + filter_H):
                        for x in range(in_x, in_x + filter_W):
                            if (x < 0 or x >= W or y < 0 or y >= H):
                                Y[batch_index, row_idx, col_index] = 0
                            else:
                                der_X[batch_index, c, y, x] += Y[batch_index, row_idx, col_index] 
                            row_idx += 1
        return der_X

    def np_conv2d_transpose(X, Filter, Y, padding = 0, stride = 1):
        """Implement a conv2d_transpose as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        X_N, X_C, X_H, X_W = X.shape
        Y_N, Y_C, Y_H, Y_W = Y.shape
        YY = Y.reshape((Y_N, Y_C, Y_H * Y_W))    # transformed to im2col Y
        # XX = X.reshape((X_N, X_C, X_W * X_H))   # transformed to im2col X
        F_filter = Filter.reshape((filter_outChannel,-1))
        gradient_im2col_XX = np.matmul(F_filter.T, YY) 
        
        gradient_X = im2col_transpose(X, filter_H, filter_W, gradient_im2col_XX,padding,stride)    # gradient of x
        im2col_XX = im2col(X, filter_H, filter_W, padding, stride)
        gradient_filter = np.zeros(shape = F_filter.shape, dtype = X.dtype)

        for i in range(X_N):
          gradient_filter += np.matmul(YY[i],im2col_XX[i].T)
        gradient_filter = gradient_filter.reshape(Filter.shape)

        return gradient_X, gradient_filter


    shapeX = (100, 3, 28, 28)
    shapeF = (10, 3, 5, 5)
    shapeY = (100, 10, 24, 24)
    shapeW = (100, 3*5*5, 24*24)
    shapeFF = (100, 10, 3, 5, 5)
    #  input : x , filter : f , output: y
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    f = np.random.uniform(0, 10, size=shapeF).astype(np.float32)  

    der_y = np.ones(shape = shapeY)
    gradient_x, gradient_f = np_conv2d_transpose(x, f, der_y)
    
    arr_x = ndarray.array(x, ctx=ctx)
    arr_f = ndarray.array(f, ctx=ctx)
    gradient_y = ndarray.array(der_y, ctx = ctx)
    gradient_xx = ndarray.array(x, ctx = ctx)
    gradient_ff = ndarray.array(f, ctx = ctx)

    # arr_y = ndarray.empty(shapeY, ctx=ctx)
    arr_workspace_im2col = ndarray.empty(shapeW, ctx=ctx)
    arr_workspace_batch_filter = ndarray.empty(shapeFF, ctx = ctx)
    # gpu_op.conv2d_transpose(arr_x, arr_f, gradient_y, gradient_xx, gradient_ff, arr_workspace_im2col, arr_workspace_batch_filter)
    gpu_op.conv2d_gradient_of_filter(arr_x, gradient_y, gradient_ff, arr_workspace_im2col, arr_workspace_batch_filter)
    gpu_op.conv2d_gradient_of_data(arr_f, gradient_y, gradient_xx, arr_workspace_im2col)

    np.testing.assert_allclose(gradient_x, gradient_xx.asnumpy(), rtol=1e-5)
    #test ok
    np.testing.assert_allclose(gradient_f, gradient_ff.asnumpy(), rtol=1e-5)


def test_average_pooling():
    ctx = ndarray.gpu(0)
    def np_average_pooling(input, kernel_H, kernel_W, padding = 0, stride = 1):
        N, C, H, W = input.shape
        assert((H + 2 * padding - kernel_H)%stride == 0)
        assert((W + 2 * padding - kernel_W)%stride == 0)
        pooled_H = (H + 2 * padding - kernel_H) // stride + 1
        pooled_W = (W + 2 * padding - kernel_W) // stride + 1
        pooled_layer = np.zeros(shape = ( N , C, pooled_H, pooled_W), dtype = np.float32)
        pooling_size = kernel_H * kernel_W
        for n in range(N):
            for c in range(C):
                for h in range(pooled_H):
                    for w in range(pooled_W):
                      hs = h * stride - padding
                      ws = w * stride - padding
                      hend = min(hs + kernel_H, H)
                      wend = min(ws + kernel_W, W)
                      hs = max(hs, 0)
                      ws = max(ws, 0)
                      for i in range(hs, hend):
                          for j in range(ws, wend):
                            pooled_layer[n][c][h][w] += input[n][c][i][j]
                      pooled_layer[n][c][h][w] /= pooling_size
        return pooled_layer
    def np_average_pooling_gradient(gradient_y, kernel_H, kernel_W, padding = 0, stride = 1):
        N, C , pooled_H, pooled_W = gradient_y.shape
        H = (pooled_H - 1) * stride + kernel_H - 2 * padding
        W = (pooled_W - 1) * stride + kernel_W - 2 * padding
        
        gradient_x = np.zeros(shape = (N, C, H, W), dtype = np.float32)
        pooling_size = kernel_H * kernel_W
        for n in range(N):
            for c in range(C):
                for h in range(pooled_H):
                    for w in range(pooled_W):
                      hs = h * stride - padding
                      ws = w * stride - padding
                      hend = min(hs + kernel_H, H) 
                      wend = min(ws + kernel_W, W)
                      hs = max(hs, 0)
                      ws = max(ws, 0)
                      for i in range(hs, hend):
                          for j in range(ws, wend):
                            gradient_x[n][c][i][j] += gradient_y[n][c][h][w] / pooling_size

        return gradient_x

    shapeX = (100, 3, 28, 28)
    # (1,1,5,5)
    shapeY = (100, 3, 24, 24)
    #  input : x , filter : f , output: y
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32) 
    gradient_y = np.random.uniform(0, 10, size=shapeY).astype(np.float32) 

    arr_x = ndarray.array(x, ctx = ctx)
    arr_gradient_y = ndarray.array(gradient_y, ctx = ctx)
    arr_pool_layer = ndarray.empty(shapeY, ctx = ctx)
    arr_gradient_x = ndarray.empty(shapeX, ctx = ctx)

    gpu_op.average_pooling2d(arr_x, 5,5,arr_pool_layer)
    gpu_op.average_pooling2d_gradient(arr_gradient_y, 5, 5, arr_gradient_x)

    np_pool_layer = np_average_pooling(x, 5, 5)
    np_gradient_x = np_average_pooling_gradient(gradient_y, 5, 5)

    np.testing.assert_allclose(np_pool_layer, arr_pool_layer.asnumpy(), rtol=1e-5)

    np.testing.assert_allclose(np_gradient_x, arr_gradient_x.asnumpy(), rtol=1e-5)


def test_reshape():
    ctx = ndarray.gpu(0)
    def np_reshape(X, output_shape):
      return X.reshape(output_shape)
    
    
    shapeX = (10, 5 ,28, 28)
    shapeY = (50, 28, 28)
    
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    y = np_reshape(x, shapeY)

    arr_x = ndarray.array(x, ctx = ctx)
    arr_y = ndarray.empty(shapeY, ctx = ctx)
    gpu_op.array_reshape(arr_x, arr_y)
    np.testing.assert_allclose(y, arr_y.asnumpy(), rtol = 1e-5)


def test_conv2d_broadcast_to():
    ctx = ndarray.gpu(0)
    shapeX = (32)
    shapeY = (100, 32, 28, 28)
    shapeW = (100, 28, 28, 32)
    x = np.random.uniform(0, 10, size = shapeX).astype(np.float32)
    np_y = np.broadcast_to(x, shapeW)
    np_y = np_y.swapaxes(1,3)

    arr_x = ndarray.array(x, ctx = ctx)
    arr_y = ndarray.empty(shapeY, ctx = ctx)
    gpu_op.conv2d_broadcast_to(arr_x,arr_y)

    np.testing.assert_allclose(np_y, arr_y.asnumpy(), rtol = 1e-5)


def test_conv2d_reduce_sum():
    ctx = ndarray.gpu(0)
    shapeX = (32,)
    shapeY = (100, 32, 28, 28)
    shapeW = (100, 28, 28, 32)
    x = np.random.uniform(0, 10, size = shapeY).astype(np.float32)
    np_y = np.sum(x, axis = (0, 2, 3))

    arr_x = ndarray.array(x, ctx = ctx)
    arr_y = ndarray.empty(shapeX, ctx = ctx)

    gpu_op.conv2d_reduce_sum(arr_x, arr_y)
    np.testing.assert_allclose(np_y, arr_y.asnumpy(), rtol = 1e-5)


def test_cudnn_conv2d():
    ctx = ndarray.gpu(0)
    # im2col and np_conv2d are helper functions
    def im2col(X, filter_H, filter_W, padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1

        y_row_size = C * filter_H * filter_W
        y_col_size = out_H * out_W
        y_shape = (N, y_row_size, y_col_size)
        Y = np.empty(y_shape, dtype = X.dtype)

        for batch_index in range(N):
            for col_index in range(y_col_size):
              out_y = col_index // out_W
              out_x = col_index % out_W
              in_y = out_y * stride - padding
              in_x = out_x * stride - padding
              row_idx = 0
              for c in range(0, C):
                  for y in range(in_y, in_y + filter_H):
                      for x in range(in_x, in_x + filter_W):
                          if (x < 0 or x >= W or y < 0 or y >= H):
                              Y[batch_index, row_idx, col_index] = 0
                          else:
                              Y[batch_index, row_idx, col_index] = X[batch_index, c, y, x]
                          row_idx += 1
        return Y

    def np_conv2d(X, Filter, padding=0, stride=1):
        """Implement a conv2d as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1

        im2col_matrix = im2col(X, filter_H, filter_W, padding, stride)
        filter_matrix = Filter.reshape(filter_outChannel, -1)
        print("shape", im2col_matrix.shape)
        print("shape", filter_matrix.shape)
        print("shape", np.matmul(filter_matrix, im2col_matrix).shape)
        return np.matmul(filter_matrix, im2col_matrix).reshape(N, filter_outChannel, out_H, out_W)
        #return im2col_matrix

    shapeX = (100, 3, 28, 28)
    shapeF = (10, 3, 5, 5)
    shapeY = (100, 10, 24, 24)
    shapeW = (100, 3*5*5, 24*24)
    # shapeX = (1,1,5,5)
    # shapeF = (1,1,1,1)
    # shapeY = (1,1,5,5)
    # shapeW = (1,1,25)
    # x = np.ones(shape = shapeX).astype(np.float32)
    # f = np.ones(shape = shapeF).astype(np.float32)
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    f = np.random.uniform(0, 10, size=shapeF).astype(np.float32)
    # x = np.arange(1,5*5+1)
    # f = np.arange(1,2)
    # # f =f*2
    # x = x.reshape(shapeX)
    # f = f.reshape(shapeF)
    y = np.zeros(shapeY).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_f = ndarray.array(f, ctx=ctx)
    arr_y = ndarray.empty(shapeY, ctx=ctx)
    arr_workspace = ndarray.empty(shapeW, ctx=ctx)

    gpu_op.CuDNN_conv2d(arr_x, arr_f, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np_conv2d(x, f), y, rtol=1e-5)


def test_cudnn_conv2d_Gradient():
    ctx = ndarray.gpu(0)
    def im2col(X, filter_H, filter_W, padding, stride):
      N, C, H, W = X.shape
      assert (H + 2 * padding - filter_H) % stride == 0
      assert (W + 2 * padding - filter_W) % stride == 0
      out_H = (H + 2 * padding - filter_H) // stride + 1
      out_W = (W + 2 * padding - filter_W) // stride + 1

      y_row_size = C * filter_H * filter_W
      y_col_size = out_H * out_W
      y_shape = (N, y_row_size, y_col_size)
      Y = np.empty(y_shape, dtype = X.dtype)

      for batch_index in range(N):
          for col_index in range(y_col_size):
            out_y = col_index // out_W
            out_x = col_index % out_W
            in_y = out_y * stride - padding
            in_x = out_x * stride - padding
            row_idx = 0
            for c in range(0, C):
                for y in range(in_y, in_y + filter_H):
                    for x in range(in_x, in_x + filter_W):
                        if (x < 0 or x >= W or y < 0 or y >= H):
                            Y[batch_index, row_idx, col_index] = 0
                        else:
                            Y[batch_index, row_idx, col_index] = X[batch_index, c, y, x]
                        row_idx += 1
      return Y
    def np_conv2d(X, Filter, padding=0, stride=1):
        """Implement a conv2d as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1

        im2col_matrix = im2col(X, filter_H, filter_W, padding, stride)
        filter_matrix = Filter.reshape(filter_outChannel, -1)
        # print("shape", im2col_matrix.shape)
        # print("shape", filter_matrix.shape)
        # print("shape", np.matmul(filter_matrix, im2col_matrix).shape)
        return np.matmul(filter_matrix, im2col_matrix).reshape(N, filter_outChannel, out_H, out_W)
        #return im2col_matrix
    def im2col_transpose(X, filter_H, filter_W, Y , padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1
        _, y_row_size, y_col_size = Y.shape

        der_X_shape = (N, C, H, W)
        der_X = np.zeros(der_X_shape, dtype = X.dtype)


        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index // out_W
                out_x = col_index % out_W
                in_y = out_y * stride - padding
                in_x = out_x * stride - padding
                row_idx = 0
                for c in range(0, C):
                    for y in range(in_y, in_y + filter_H):
                        for x in range(in_x, in_x + filter_W):
                            if (x < 0 or x >= W or y < 0 or y >= H):
                                Y[batch_index, row_idx, col_index] = 0
                            else:
                                der_X[batch_index, c, y, x] += Y[batch_index, row_idx, col_index] 
                            row_idx += 1
        return der_X

    def np_conv2d_transpose(X, Filter, Y, padding = 0, stride = 1):
        """Implement a conv2d_transpose as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        X_N, X_C, X_H, X_W = X.shape
        Y_N, Y_C, Y_H, Y_W = Y.shape
        YY = Y.reshape((Y_N, Y_C, Y_H * Y_W))    # transformed to im2col Y
        # XX = X.reshape((X_N, X_C, X_W * X_H))   # transformed to im2col X
        F_filter = Filter.reshape((filter_outChannel,-1))
        gradient_im2col_XX = np.matmul(F_filter.T, YY) 
        
        gradient_X = im2col_transpose(X, filter_H, filter_W, gradient_im2col_XX,padding,stride)    # gradient of x
        im2col_XX = im2col(X, filter_H, filter_W, padding, stride)
        gradient_filter = np.zeros(shape = F_filter.shape, dtype = X.dtype)

        for i in range(X_N):
          gradient_filter += np.matmul(YY[i],im2col_XX[i].T)
        gradient_filter = gradient_filter.reshape(Filter.shape)

        return gradient_X, gradient_filter


    shapeX = (100, 3, 28, 28)
    shapeF = (10, 3, 5, 5)
    shapeY = (100, 10, 24, 24)
    shapeW = (100, 3*5*5, 24*24)
    shapeFF = (100, 10, 3, 5, 5)
    #  input : x , filter : f , output: y
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    f = np.random.uniform(0, 10, size=shapeF).astype(np.float32)  

    der_y = np.ones(shape = shapeY)
    gradient_x, gradient_f = np_conv2d_transpose(x, f, der_y)
    
    arr_x = ndarray.array(x, ctx=ctx)
    arr_f = ndarray.array(f, ctx=ctx)
    gradient_y = ndarray.array(der_y, ctx = ctx)
    gradient_xx = ndarray.array(x, ctx = ctx)
    gradient_ff = ndarray.array(f, ctx = ctx)

    # arr_y = ndarray.empty(shapeY, ctx=ctx)
    arr_workspace_im2col = ndarray.empty(shapeW, ctx=ctx)
    arr_workspace_batch_filter = ndarray.empty(shapeFF, ctx = ctx)
    # gpu_op.conv2d_transpose(arr_x, arr_f, gradient_y, gradient_xx, gradient_ff, arr_workspace_im2col, arr_workspace_batch_filter)
    gpu_op.CuDNN_conv2d_gradient_of_filter(arr_x, gradient_y, gradient_ff)
    gpu_op.CuDNN_conv2d_gradient_of_data(arr_f, gradient_y, gradient_xx)

    np.testing.assert_allclose(gradient_x, gradient_xx.asnumpy(), rtol=1e-5)
    #test ok
    np.testing.assert_allclose(gradient_f, gradient_ff.asnumpy(), rtol=1e-5)


def test_average_pooling():
    ctx = ndarray.gpu(0)
    def np_average_pooling(input, kernel_H, kernel_W, padding = 0, stride = 1):
        N, C, H, W = input.shape
        assert((H + 2 * padding - kernel_H)%stride == 0)
        assert((W + 2 * padding - kernel_W)%stride == 0)
        pooled_H = (H + 2 * padding - kernel_H) // stride + 1
        pooled_W = (W + 2 * padding - kernel_W) // stride + 1
        pooled_layer = np.zeros(shape = ( N , C, pooled_H, pooled_W), dtype = np.float32)
        pooling_size = kernel_H * kernel_W
        for n in range(N):
            for c in range(C):
                for h in range(pooled_H):
                    for w in range(pooled_W):
                      hs = h * stride - padding
                      ws = w * stride - padding
                      hend = min(hs + kernel_H, H)
                      wend = min(ws + kernel_W, W)
                      hs = max(hs, 0)
                      ws = max(ws, 0)
                      for i in range(hs, hend):
                          for j in range(ws, wend):
                            pooled_layer[n][c][h][w] += input[n][c][i][j]
                      pooled_layer[n][c][h][w] /= pooling_size
        return pooled_layer
    def np_average_pooling_gradient(gradient_y, kernel_H, kernel_W, padding = 0, stride = 1):
        N, C , pooled_H, pooled_W = gradient_y.shape
        H = (pooled_H - 1) * stride + kernel_H - 2 * padding
        W = (pooled_W - 1) * stride + kernel_W - 2 * padding
        
        gradient_x = np.zeros(shape = (N, C, H, W), dtype = np.float32)
        pooling_size = kernel_H * kernel_W
        for n in range(N):
            for c in range(C):
                for h in range(pooled_H):
                    for w in range(pooled_W):
                      hs = h * stride - padding
                      ws = w * stride - padding
                      hend = min(hs + kernel_H, H) 
                      wend = min(ws + kernel_W, W)
                      hs = max(hs, 0)
                      ws = max(ws, 0)
                      for i in range(hs, hend):
                          for j in range(ws, wend):
                            gradient_x[n][c][i][j] += gradient_y[n][c][h][w] / pooling_size

        return gradient_x

    shapeX = (100, 3, 28, 28)
    # # (1,1,5,5)
    shapeY = (100, 3, 24, 24)
    # shapeX = (1, 1, 6, 6)
    # # (1,1,5,5)
    # shapeY = (1, 1, 5, 5)
    #  input : x , filter : f , output: y
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32) 
    # x = np.arange(1,37).reshape(shapeX)
    # print(x)
    # x = np.ones(shapeX).astype(np.float32)
    gradient_y = np.random.uniform(0, 10, size=shapeY).astype(np.float32) 
    arr_x = ndarray.array(x, ctx = ctx)
    arr_gradient_y = ndarray.array(gradient_y, ctx = ctx)
    arr_pool_layer = ndarray.empty(shapeY, ctx = ctx)
    arr_gradient_x = ndarray.empty(shapeX, ctx = ctx)

    gpu_op.CuDNN_average_pooling2d(arr_x, 5, 5,arr_pool_layer)
    gpu_op.CuDNN_average_pooling2d_gradient(arr_pool_layer, arr_gradient_y, arr_x, 5, 5, arr_gradient_x)
    # print(arr_pool_layer.asnumpy())
    np_pool_layer = np_average_pooling(x, 5, 5)
    # print(np_pool_layer)
    np_gradient_x = np_average_pooling_gradient(gradient_y, 5, 5)

    np.testing.assert_allclose(np_pool_layer, arr_pool_layer.asnumpy(), rtol=1e-5)

    np.testing.assert_allclose(np_gradient_x, arr_gradient_x.asnumpy(), rtol=1e-5)


def test_CuDNN_max_pooling():
    ctx = ndarray.gpu(0)
    def np_max_pooling_gradient(input, gradient_y, kernel_H, kernel_W, padding = 0, stride = 1):
        N, C , pooled_H, pooled_W = gradient_y.shape
        H = (pooled_H - 1) * stride + kernel_H - 2 * padding
        W = (pooled_W - 1) * stride + kernel_W - 2 * padding
        # print(N,C,H,W)
        gradient_x = np.zeros(shape = (N, C, H, W), dtype = np.float32)
        pooling_size = kernel_H * kernel_W

        for n in range(N):
            for c in range(C):
                for h in range(pooled_H):
                    for w in range(pooled_W):
                        hs = h * stride - padding
                        ws = w * stride - padding
                        hend = min(hs + kernel_H, H) 
                        wend = min(ws + kernel_W, W)
                        hs = max(hs, 0)
                        ws = max(ws, 0)

                        hargmax = hs
                        wargmax = ws
                        for i in range(hs, hend):
                            for j in range(ws, wend):
                                # print(n,c,i,j)
                                if input[n][c][i][j] > input[n][c][hargmax][wargmax]:
                                    hargmax = i
                                    wargmax = j
                        gradient_x[n][c][hargmax][wargmax] += gradient_y[n][c][h][w]

        return gradient_x

    shapeX = (100, 3, 28, 28)
    # # (1,1,5,5)
    shapeY = (100, 3, 24, 24)
    # shapeX = (1, 1, 6, 6)
    # # (1,1,5,5)
    # shapeY = (1, 1, 5, 5)
    #  input : x , filter : f , output: y
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32) 
    # x = np.arange(1,37).reshape(shapeX)
    # print(x)
    # x = np.ones(shapeX).astype(np.float32)
    gradient_y = np.random.uniform(0, 10, size=shapeY).astype(np.float32)
    # gradient_y = np.ones(shapeY).astype(np.float32) 
    arr_x = ndarray.array(x, ctx = ctx)
    arr_gradient_y = ndarray.array(gradient_y, ctx = ctx)
    arr_pool_layer = ndarray.empty(shapeY, ctx = ctx)
    arr_gradient_x = ndarray.empty(shapeX, ctx = ctx)

    arr_pool_layer1 = ndarray.empty(shapeY, ctx = ctx)
    arr_gradient_x1 = ndarray.empty(shapeX, ctx = ctx)

    gpu_op.CuDNN_max_pooling2d(arr_x, 2, 2,arr_pool_layer)
    gpu_op.CuDNN_max_pooling2d_gradient(arr_pool_layer, arr_gradient_y, arr_x, 2, 2, arr_gradient_x)

    gpu_op.max_pooling2d(arr_x, 2, 2, arr_pool_layer1)
    gpu_op.max_pooling2d_gradient(arr_x, arr_gradient_y, 2, 2, arr_gradient_x1)


    # print(arr_pool_layer.asnumpy())
    # np_pool_layer = np_average_pooling(x, 5, 5)
    # print(np_pool_layer)
    # np_gradient_x = np_average_pooling_gradient(gradient_y, 5, 5)

    # np.testing.assert_allclose(np_pool_layer, arr_pool_layer.asnumpy(), rtol=1e-5)

    # np.testing.assert_allclose(np_gradient_x, arr_gradient_x.asnumpy(), rtol=1e-5)


def test_CuDNN_dropout_op():
    import ctypes
    ctx = ndarray.gpu(0)
    shapeX = (20,30,50,70)
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32) 
    arr_x = ndarray.array(x, ctx = ctx)
    arr_y = ndarray.empty(shapeX, ctx = ctx)
    shapeK = (1)
    keep_prob = 0.5
    reserve_size = ctypes.c_int(0)
    reserve_space = ctypes.c_void_p(0)
    gpu_op.CuDNN_Dropout(arr_x, keep_prob, arr_y, reserve_size, reserve_space, 1)

    gradient_y = np.random.uniform(0, 10, size = shapeX).astype(np.float32)
    gradient_y = ndarray.array(gradient_y, ctx = ctx)
    gradient_x = ndarray.empty(shapeX, ctx = ctx)
    gpu_op.CuDNN_Dropout_gradient(gradient_y, keep_prob, gradient_x, reserve_size, reserve_space)
    # print(arr_y.asnumpy())
    # print(gradient_x.asnumpy())


def test_pad():
    ctx = ndarray.gpu(0)
    shape = (1, 1, 1,3)
    paddings = [[1,1],[1,1]]
    to_shape = (1,1,3,5)
    x = np.random.uniform(0, 10, size = shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx = ctx)
    arr_y = ndarray.empty(to_shape, ctx = ctx)
    gpu_op.pad(arr_x, arr_y, paddings)
    print(arr_x.asnumpy())
    print(arr_y.asnumpy())
    gradient_y = np.random.uniform(0, 10, size = to_shape).astype(np.float32)
    arr_gradient_y = ndarray.array(gradient_y, ctx = ctx)
    arr_gradient_x = ndarray.empty(shape, ctx = ctx)
    gpu_op.pad_gradient(arr_gradient_y, arr_gradient_x, paddings)
    print(arr_gradient_y.asnumpy())
    print(arr_gradient_x.asnumpy())


def test_concat():
    def unit_test(shape1, shape2, axis):
        ctx = ndarray.gpu(0)
        x1 = np.random.random(shape1).astype(np.float32)
        x2 = np.random.random(shape2).astype(np.float32)
        arr_x1 = ndarray.array(x1, ctx = ctx)
        arr_x2 = ndarray.array(x2, ctx = ctx)
        np_res = np.concatenate([x1, x2], axis)
        arr_res = ndarray.empty(np_res.shape, ctx=ctx)
        gpu_op.concat(arr_x1, arr_x2, arr_res, axis)
        np.testing.assert_allclose(arr_res.asnumpy(), np_res)

        grad_x1 = ndarray.empty(shape1, ctx = ctx)
        grad_x2 = ndarray.empty(shape2, ctx = ctx)
        gpu_op.concat_gradient(arr_res, grad_x1, axis = axis, idx = 0)
        gpu_op.concat_gradient(arr_res, grad_x2, axis = axis, idx = 1)
        np.testing.assert_allclose(x1, grad_x1.asnumpy())
        np.testing.assert_allclose(x2, grad_x2.asnumpy())

        print("Pass test with ", shape1, shape2, axis)
    
    unit_test((1, 2), (1, 2), 0)
    unit_test((12, 34, 56), (12, 43, 56), 1)
    


def test_matrix_transpose():
    shape = (4321, 1234)
    ctx = ndarray.gpu(0)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty((shape[1], shape[0]), ctx=ctx)
    gpu_op.matrix_transpose(arr_x, arr_y, perm=[1, 0])
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np.transpose(x), y)

    shape = (21, 43, 65, 11)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty((65, 11, 43, 21), ctx=ctx)
    gpu_op.matrix_transpose(arr_x, arr_y, perm=[2, 3, 1, 0])
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np.transpose(x, [2, 3, 1, 0]), y)


def test_slice():
    i_shape = (123, 234, 13, 7)
    o_shape = (67, 209, 3, 5)
    begin_pos = (31, 11, 3, 1)
    
    ctx = ndarray.gpu(0)
    x = np.random.normal(size=i_shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(o_shape, ctx=ctx)
    
    gpu_op.matrix_slice(arr_x, arr_y, begin_pos)
    y = arr_y.asnumpy()
    index = tuple([slice(i, i+j) for i, j in zip(begin_pos, o_shape)])
    np.testing.assert_allclose(x[index], y)
    print("Slice op no bug.")

    begin_pos = (29, 3, 5, 0)
    gpu_op.matrix_slice_gradient(arr_y, arr_x, begin_pos)
    x = arr_x.asnumpy()
    index = tuple([slice(i, i+j) for i, j in zip(begin_pos, o_shape)])
    x_ = np.zeros(i_shape, dtype=np.float32)
    x_[index] = y
    np.testing.assert_allclose(x, x_)
    print("Slice gradient op no bug.")


def test_embedding_lookup():
    '''
    emb = np.random.rand(5, 5)
    ctx = ndarray.gpu(0)
    print(emb)
    emb = ndarray.array(emb, ctx = ctx)
    ids = [0, 1, 0]
    ids = np.array(ids)
    print(ids)
    ids = ndarray.array(ids, ctx = ctx)

    output = ndarray.empty((3, 5), ctx = ctx)
    gpu_op.embedding_lookup(emb, ids, output)
    print(output.asnumpy())

    grad_emb = ndarray.empty((5,5), ctx=ctx)
    gpu_op.embedding_lookup_gradient(output, ids, grad_emb)
    print(grad_emb.asnumpy())
    # numpy
    print("Testing NumPy")
    emb = np.random.rand(5, 5)
    ids = [0, 1, 0]
    ids = np.array(ids)
    print(emb)
    print(ids)
    print(emb[ids])
    '''
    emb = np.random.rand(5, 5)
    ctx = ndarray.gpu(0)
    print(emb)
    emb = ndarray.array(emb, ctx = ctx)
    ids = [[0, 1], [0, 1]]
    ids = np.array(ids)
    print(ids)
    ids = ndarray.array(ids, ctx = ctx)

    output = ndarray.empty((2, 2, 5), ctx = ctx)
    gpu_op.embedding_lookup(emb, ids, output)
    print(output.asnumpy())

    grad_emb = ndarray.empty((5,5), ctx=ctx)
    gpu_op.embedding_lookup_gradient(output, ids, grad_emb)
    print(grad_emb.asnumpy())
    # numpy
    print("Testing NumPy")
    emb = np.random.rand(5, 5)
    ids = [[0, 1], [0, 1]]
    ids = np.array(ids)
    flatten_ids = ids.reshape(-1)
    print(emb)
    print(flatten_ids)
    print(emb[flatten_ids].reshape(ids.shape[0], ids.shape[1], emb.shape[1]))
    
    # output = ndarray.empty((3, 5), ctx = ctx)
    # gpu_op.embedding_lookup(emb, ids, output)
    # print(output.asnumpy())

    # grad_emb = ndarray.empty((5,5), ctx=ctx)
    # gpu_op.embedding_lookup_gradient(output, ids, grad_emb)
    # print(grad_emb.asnumpy())


def test_where():
    cond = np.random.randint(2, size=(5, 5)).astype(np.float32)
    x = np.random.rand(5, 5).astype(np.float32)
    y = np.random.rand(5, 5).astype(np.float32)
    ctx = ndarray.gpu(0)
    cond_gpu = ndarray.array(cond, ctx=ctx)
    x_gpu = ndarray.array(x, ctx=ctx)
    y_gpu = ndarray.array(y, ctx=ctx)
    output_gpu = ndarray.empty((5, 5), ctx=ctx)

    gpu_op.where(cond_gpu, x_gpu, y_gpu, output_gpu)
    output = np.where(cond, x, y)

    # assert((output == output_gpu.asnumpy()).all())
    np.testing.assert_allclose(output, output_gpu.asnumpy())


def test_batch_matrix_multiply():
    ctx = ndarray.gpu(0)
    x = np.random.uniform(0, 10, size=(2, 3, 500, 700)).astype(np.float32)
    y = np.random.uniform(0, 10, size=(2, 3, 700, 1000)).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty((2, 3, 500, 1000), ctx=ctx)
    gpu_op.batch_matrix_multiply(arr_x, False, arr_y, False, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.matmul(x, y), z, rtol=1e-5)
    print('Test 1 passed.')

    x = np.random.uniform(0, 10, size=(7, 11, 1000, 500)).astype(np.float32)
    y = np.random.uniform(0, 10, size=(7, 11, 2000, 500)).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty((7, 11, 1000, 2000), ctx=ctx)
    gpu_op.batch_matrix_multiply(arr_x, False, arr_y, True, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.matmul(x, np.transpose(y, [0,1,3,2])), z, rtol=1e-5)
    print('Test 2 passed.')
    
    x = np.random.uniform(0, 10, size=(3, 2, 5, 500, 1000)).astype(np.float32)
    y = np.random.uniform(0, 10, size=(3, 2, 5, 2000, 500)).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty((3, 2, 5, 1000, 2000), ctx=ctx)
    gpu_op.batch_matrix_multiply(arr_x, True, arr_y, True, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.matmul(np.transpose(x, [0,1,2,4,3]), np.transpose(y, [0,1,2,4,3])), z,
                               rtol=1e-5)
    print('Test 3 passed.')


def test_broadcast_shape():
    def unit_test(shape1, shape2):
        ctx = ndarray.gpu(0)
        x = np.random.random(shape1).astype(np.float32)
        arr_x = ndarray.array(x, ctx=ctx)
        arr_y = ndarray.empty(shape2, ctx=ctx)
        gpu_op.broadcast_shape(arr_x, arr_y)
        np.testing.assert_allclose(arr_y.asnumpy(), np.broadcast_to(x, shape2))
        print('Passed test with input shape %s and output shape %s.' % (str(shape1), str(shape2)))
    
    unit_test((3, 1), (2, 3, 4))
    unit_test((1,), (2, 3, 4, 5))
    unit_test((1, 1, 3, 1), (9, 8, 3, 7))


def test_reduce_sum():
    def unit_test(shape, axes):
        ctx = ndarray.gpu(0)
        x = np.random.random(shape).astype(np.float32)
        arr_x = ndarray.array(x, ctx=ctx)
        o_shape = list(shape)
        for ax in axes:
            o_shape[ax] = 0
        o_shape = [i for i in o_shape if i > 0]
        arr_y = ndarray.empty(o_shape, ctx=ctx)
        gpu_op.reduce_sum(arr_x, arr_y, axes)
        np.testing.assert_allclose(arr_y.asnumpy(), np.sum(x, tuple(axes), keepdims=False), rtol=1e-6)
        print('Passed test with input shape %s and reduce axes %s.' % (str(shape), str(axes)))
    
    unit_test((2, 3, 4), [2])
    unit_test((2, 3, 4), [2, 1])
    unit_test((2, 3, 4), [2, 1, 0])
    unit_test((2, 3, 1, 5, 6), [1, 2, 4])


def test_reduce_mean():
    def unit_test(shape, axes):
        ctx = ndarray.gpu(0)
        x = np.random.random(shape).astype(np.float32)
        arr_x = ndarray.array(x, ctx=ctx)
        o_shape = list(shape)
        for ax in axes:
            o_shape[ax] = 0
        o_shape = [i for i in o_shape if i > 0]
        arr_y = ndarray.empty(o_shape, ctx=ctx)
        gpu_op.reduce_mean(arr_x, arr_y, axes)
        np.testing.assert_allclose(arr_y.asnumpy(), np.mean(x, tuple(axes), keepdims=False), rtol=1e-6)
        print('Passed test with input shape %s and reduce axes %s.' % (str(shape), str(axes)))
    
    unit_test((2, 3, 4), [2])
    unit_test((2, 3, 4), [2, 1])
    unit_test((2, 3, 4), [2, 1, 0])
    unit_test((2, 3, 1, 5, 6), [1, 2, 4])


def test_dropout():
    ctx = ndarray.gpu(0)
    # shapeX = (16, 8, 99, 64)
    shapeX = (10,)
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32) 
    arr_x = ndarray.array(x, ctx = ctx)
    arr_y = ndarray.empty(shapeX, ctx = ctx)
    dropout_rate = 0.6
    import ctypes
    seed = ctypes.c_ulonglong(0)
    print(x)
    gpu_op.dropout(arr_x, dropout_rate, arr_y, seed)
    print(arr_y.asnumpy())
    print(seed)
    gpu_op.dropout_gradient(arr_x, dropout_rate, arr_y, seed)
    print(arr_y.asnumpy())
    print(seed)


def test_onehot():
    ctx = ndarray.gpu(0)
    shapein = (2, 3)
    num_classes = 7
    x = np.random.randint(7, size=shapein)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(list(shapein) + [num_classes], ctx=ctx)
    gpu_op.one_hot(arr_x, arr_y)
    print(x)
    print(arr_y.asnumpy())


test_array_set()
test_broadcast_to()
test_reduce_sum_axis_zero()
test_matrix_elementwise_add()
test_matrix_elementwise_add_by_const()
test_matrix_elementwise_multiply()
test_matrix_elementwise_multiply_by_const()
test_matrix_elementwise_divide()
test_matrix_elementwise_divide_const()
test_matrix_opposite()
test_matrix_multiply()
test_matrix_sqrt()
test_matrix_rsqrt()
test_relu()
test_relu_gradient()
test_softmax()
test_softmax_cross_entropy()
test_softmax_cross_entropy_gradient()
test_conv2d()
test_conv2d_Gradient()
test_average_pooling()
test_reshape()
test_conv2d_broadcast_to()
test_conv2d_reduce_sum()
test_cudnn_conv2d()
test_cudnn_conv2d_Gradient()
test_average_pooling()
test_CuDNN_max_pooling()
test_CuDNN_dropout_op()
test_pad()
test_concat()
test_matrix_transpose()
test_slice()
test_embedding_lookup()
test_where()
test_batch_matrix_multiply()
test_broadcast_shape()
test_reduce_sum()
test_reduce_mean()
test_dropout()
test_onehot()