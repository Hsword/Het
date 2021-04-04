# gpu_ops
This directory contains executor and operators for computation and communication. Though the name of directory is "gpu_ops", in each operator we call different API for computation in NumPy(CPU), DNNL(CPU), CUDA(GPU) according to the context specified in executor and the environment.

## Executor
* Defined in executor.py, contains all the configurations and controls the training/inference process.

## Operators
### Computation
| Operator | NumPy(CPU) | DNNL(CPU) | CUDA(GPU) | CUDA Backend |
| :----: | :----: | :----: | :----: | :----: |
| AddByConstOp | ✔ | ✔ | ✔ | / |
| AddOp | ✔ | ✔ | ✔ | / |
| Avg_Pool2dOp | ✔ | ✔ | ✔ | CuDNN |
| Avg_Pool2d_GradientOp | ✔ | ✔ | ✔ | CuDNN |
| BatchMatMulOp | ✔ | ✖ | ✔ | CuBLAS |
| Batch_NormalizationOp | ✔ | ✔ | ✔ | CuDNN |
| Batch_Normalization_GradientOp | ✔ | ✔ | ✔ | CuDNN |
| BinaryCrossEntropyOp | ✔ | ✖ | ✔ | / |
| BroadcastToOp | ✔ | ✖ | ✔ | / |
| BroadcastShapeOp | ✔ | ✖ | ✔ | / |
| ConcatOp | ✔ | ✔ | ✔ | / |
| Concat_gradientOP | ✔ | ✔ | ✔ | / |
| Conv2dOp | ✔ | ✔ | ✔ | / |
| Conv2d_Gradient_of_DataOp | ✔ | ✔ | ✔ | / |
| Conv2d_Gradient_of_FilterOp | ✔ | ✔ | ✔ | / |
| Conv2d_BroadcastToOp | ✔ | ✖ | ✔ | / |
| Conv2d_ReduceSumOp | ✔ | ✖ | ✔ | / |
| CsrmvOp | ✔ | ✖ | ✔ | / |
| CsrmmOp | ✔ | ✖ | ✔ | / |
| DivOp | ✔ | ✔ | ✔ | / |
| DivConstOp | ✔ | ✔ | ✔ | / |
| DropoutOp | ✔ | ✔ | ✔ | CuRAND |
| EmbeddingLookUp | ✔ | ✖ | ✔ | / |
| EmbeddingLookUp_Gradient | ✔ | ✖ | ✔ | / |
| Layer_NormalizationOp | ✔ | ✖ | ✔ | CuDNN |
| Layer_Normalization_GradientOp | ✔ | ✖ | ✔ | CuDNN |
| MatMulOp | ✔ | ✔ | ✔ | CuBLAS |
| Max_Pool2dOp | ✔ | ✔ | ✔ | CuDNN |
| Max_Pool2d_GradientOp | ✔ | ✔ | ✔ | CuDNN |
| MulByConstOp | ✔ | ✔ | ✔ | / |
| MulOp | ✔ | ✔ | ✔ | / |
| OneHotOp | ✔ | ✖ | ✔ | / |
| OnesLikeOp | ✔ | ✔ | ✔ | / |
| OppositeOp | ✔ | ✔ | ✔ | / |
| PadOp | ✔ | ✔ | ✔ | / |
| Pad_GradientOp | ✔ | ✔ | ✔ | / |
| ReduceMeanOp | ✔ | ✖ | ✔ | CuDNN |
| ReduceSumOp | ✔ | ✖ | ✔ | CuDNN |
| ReduceSumAxisZeroOp | ✔ | ✔ | ✔ | / |
| ReluOp | ✔ | ✔ | ✔ | / |
| ReluGradientOp | ✔ | ✔ | ✔ | / |
| Array_ReshapeOp | ✔ | ✔ | ✔ | / |
| SigmoidOp | ✔ | ✔ | ✔ | / |
| SliceOp | ✔ | ✖ | ✔ | / |
| SliceGradientOp | ✔ | ✖ | ✔ | / |
| SoftmaxOp | ✔ | ✔ | ✔ | CuDNN |
| SoftmaxGradientOp | ✔ | ✖ | ✔ | CuDNN |
| SoftmaxCrossEntropyOp | ✔ | ✔ | ✔ | CuDNN (Optional) |
| SoftmaxCrossEntropyGradientOp | ✔ | ✖ | ✔ | CuDNN (Optional) |
| SqrtOp | ✔ | ✔ | ✔ | / |
| ReciprocalSqrtOp | ✔ | ✔ | ✔ | / |
| TanhOp | ✔ | ✔ | ✔ | / |
| TransposeOp | ✔ | ✔ | ✔ | / |
| WhereOp | ✔ | ✖ | ✔ | / |
| ZerosLikeOp | ✔ | ✔ | ✔ | / |
| OptimizerOp | ✔ | ✔ | ✔ | / |
| OptimizerOp for sparse | ✔ | ✖ | ✔ | / |
| DataloaderOp | ✔ | ✔ | / | / |
| MatrixDotOp | ✔ | ✖ | ✔ | / |

### Communication
* DataH2DOp
* DataD2HOp
* DataD2HSparseOp
* AllReduceCommunicateOp
* ParameterServerCommunicateOp
