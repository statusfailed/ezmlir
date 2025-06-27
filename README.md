# EZMLIR

MLIR → Host code. As easy as possible.

![MLIR](./tradeoffer.png)

You provide an MLIR file like this:

```llvm
module {
 func.func @matmul_chain(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<16x16xf32>) -> tensor<4x16xf32> {
   %0 = tensor.empty() : tensor<4x16xf32>
   %1 = linalg.matmul ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x16xf32>) outs(%0 : tensor<4x16xf32>) -> tensor<4x16xf32>
   %2 = tensor.empty() : tensor<4x16xf32>
   %3 = linalg.matmul ins(%1, %arg2 : tensor<4x16xf32>, tensor<16x16xf32>) outs(%2 : tensor<4x16xf32>) -> tensor<4x16xf32>
   return %3 : tensor<4x16xf32>
 }
}
```

Then you produce a host binary like this:

    python main.py sample.mlir # add --suffix=-20 to set tool paths to e.g. mlir-opt-20

then you will have an object file `matmul_chain.o` with a `matmul_chain` symbol in your output.

# Dependencies

This script only uses LLVM MLIR binaries; no python bindings.
You'll need the following MLIR tools in your path:

    mlir-opt
    mlir-transform

On Ubuntu 25.04 you can install these as mlir-20.

# Compiler Passes

Optimization, lowering, ...
