from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from onnx import onnx_pb
from athena.onnx import constants, util,graph
from athena.onnx.handler import athena_op
from athena.onnx.onnx_opset import general

@athena_op(["SoftmaxOp"],onnx_op=["Softmax"])
class Softmax():
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass
        # logits_rank = len(ctx.get_shape(node.input_tensor_names[0]))
        # node.set_attr("axis",logits_rank - 1)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        cls.version_1(ctx, node, **kwargs)