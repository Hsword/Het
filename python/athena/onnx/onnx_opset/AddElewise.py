from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from onnx import onnx_pb
from athena.onnx import constants, util,graph
from athena.onnx.handler import athena_op
from athena.onnx.onnx_opset import general

@athena_op(["AddOp"],onnx_op=["Add"])
class Add:
    @classmethod
    def version_1(cls,ctx,node,**kwargs):
        shape0=ctx.get_shape(node._inputs[0])
        shape1=ctx.get_shape(node._inputs[1])
        if shape0!=shape1:
            node.set_attr('broadcast',1)
            if shape0 and shape1 and len(shape0)<len(shape1):
                tmp=node._inputs[0]
                ctx.replace_input(node,node._inputs[0],node._inputs[1],0)
                ctx.replace_input(node,node._inputs[1],tmp,1)
        else:
            node.set_attr('broadcast',0)


    @classmethod
    def version_6(cls,ctx,node,**kwargs):
        shape0=ctx.get_shape(node._inputs[0])
        shape1=ctx.get_shape(node._inputs[1])
        if shape0 and shape1 and len(shape0)<len(shape1):
            tmp=node._inputs[0]
            ctx.replace_input(node,node._inputs[0],node._inputs[1],0)
            ctx.replace_input(node,node._inputs[1],tmp,1)