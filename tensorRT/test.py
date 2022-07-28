import os
import numpy as np
import tensorrt as trt
from cuda import cudart
logger = trt.Logger(trt.Logger.ERROR)

builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30

inputTensor = network.add_input('inputT0', trt.float32, [-1, -1, -1])
profile.set_shape(inputTensor.name, [1,1,1],[3,4,5],[6,8,10])  # where change
config.add_optimization_profile(profile)

identityLayer = network.add_identity(inputTensor)
network.mark_output(identityLayer.get_output(0))

engineString = builder.build_serialized_network(network, config)
if(engineString == None):
    print("Failed getting serialized engine!")
    exit()
print("Succeeded getting serialized engine!")

engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
if engine == None:
    print("Failed building engine!")
    exit()
print("success building")

context = engine.create_execution_context()
context.set_binding_shape(0, [3,4,5])  # change where

nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput
for i in range(nInput):
    print("Bind[%2d]:i[%2d]->" % (i,i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
for i in range(nInput, nInput+nOutput):
    print("Bind[%2d]:o[%2d]->" %(i, i- nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

data = np.arange(3*4*5, dtype=np.float32).reshape(3,4,5)
bufferH = []
bufferH.append(np.ascontiguousarray(data.reshape(-1)))
for i in range(nInput, nInput + nOutput):
    bufferH.append(np.empty(context.get_binding_shape(i), dtype = trt.nptype(engine.get_binding_dtype(i))))
bufferD = []
for i in range(nInput + nOutput):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

context.execute_v2(bufferD)

for i in range(nInput, nInput+nOutput):
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

for i in range(nInput+nOutput):
    print(engine.get_binding_name(i))
    print(bufferH[i].reshape(context.get_binding_shape(i)))

for b in bufferD:
    cudart.cudaFree(b)