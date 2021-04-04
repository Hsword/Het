from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from onnx import onnx_pb
from athena.onnx import constants, util,graph
from athena.onnx.handler import athena_op
from athena.onnx.onnx_opset import general

@athena_op(["SqrtOp"],onnx_op=["Sqrt"])
class Sqrt(general.PassOp):
        pass






@athena_op(["ReciprocalSqrtOp"],onnx_op=["Sqrt"])
class rSqrt:
    @classmethod
    def version_1(cls,ctx,node,**kwargs):
        op_name = util.make_name(node.name)
        reciprocal = ctx.insert_new_node_on_output(
            "Reciprocal", node.output_tensor_names[0], name=op_name
        )
        ctx.copy_shape(node.output_tensor_names[0], reciprocal.output_tensor_names[0])