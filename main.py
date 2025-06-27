#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path

class MLIRConfig:
    def __init__(self):
        self.mlir_opt = "mlir-opt"
        self.mlir_translate = "mlir-translate"
        self.llc = "llc"
        self.clang = "clang"

def optimize_mlir(input_file, output_file, config):
    """Apply optimization passes using mlir-opt"""
    cmd = [
        config.mlir_opt, input_file,
        "--canonicalize",
        "--linalg-fuse-elementwise-ops",
        "--cse",
        "--linalg-generalize-named-ops",
        "--convert-linalg-to-loops",
        "-o", output_file
    ]
    subprocess.run(cmd, check=True)

def lower_to_llvm_dialect(input_file, output_file, config):
    """Lower MLIR to LLVM dialect using mlir-opt"""
    cmd = [
        config.mlir_opt, input_file,
        "--one-shot-bufferize=bufferize-function-boundaries=1",
        "--convert-linalg-to-loops",
        "--convert-scf-to-cf",
        "--expand-strided-metadata",
        "--lower-affine",
        "--finalize-memref-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-func-to-llvm",
        "--convert-cf-to-llvm",
        "--reconcile-unrealized-casts",
        "-o", output_file
    ]
    subprocess.run(cmd, check=True)

def translate_to_llvm_ir(input_file, output_file, config):
    """Translate MLIR to LLVM IR using mlir-translate"""
    cmd = [config.mlir_translate, input_file, "--mlir-to-llvmir", "-o", output_file]
    subprocess.run(cmd, check=True)

def compile_to_object(llvm_ir_file, obj_file, config):
    """Compile LLVM IR to object file using llc"""
    cmd = [config.llc, "-filetype=obj", llvm_ir_file, "-o", obj_file]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="MLIR compiler using LLVM tools")
    parser.add_argument("input", help="Input MLIR file or '-' to read from stdin")
    parser.add_argument("--suffix", default="", help="Suffix for all LLVM binaries (e.g., '-20' for Ubuntu)")
    parser.add_argument("--mlir-opt", help="Path to mlir-opt binary (overrides --suffix)")
    parser.add_argument("--mlir-translate", help="Path to mlir-translate binary (overrides --suffix)")
    parser.add_argument("--llc", help="Path to llc binary (overrides --suffix)")
    parser.add_argument("--clang", help="Path to clang binary (overrides --suffix)")
    parser.add_argument("--keep-temps", action="store_true", help="Keep temporary files")
    parser.add_argument("--output-dir", default=".", help="Output directory for generated files")
    
    args = parser.parse_args()
    
    config = MLIRConfig()
    config.mlir_opt = args.mlir_opt or f"mlir-opt{args.suffix}"
    config.mlir_translate = args.mlir_translate or f"mlir-translate{args.suffix}"
    config.llc = args.llc or f"llc{args.suffix}"
    config.clang = args.clang or f"clang{args.suffix}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    temp_files = []

    # Read MLIR code from input
    if args.input == '-':
        mlir_code = sys.stdin.read()
    else:
        with open(args.input, 'r') as f:
            mlir_code = f.read()
    
    try:
        
        original_mlir = output_dir / "original.mlir"
        with open(original_mlir, 'w') as f:
            f.write(mlir_code)
        temp_files.append(original_mlir)
        
        print("Original MLIR:")
        print(mlir_code)
        print()
        
        # Step 2: Optimize MLIR
        print("Optimizing MLIR...")
        optimized_mlir = output_dir / "optimized.mlir"
        optimize_mlir(str(original_mlir), str(optimized_mlir), config)
        temp_files.append(optimized_mlir)
        
        with open(optimized_mlir, 'r') as f:
            optimized_code = f.read()
        print("After optimization:")
        print(optimized_code)
        print()
        
        # Step 3: Lower to LLVM dialect
        print("Lowering to LLVM dialect...")
        lowered_mlir = output_dir / "lowered.mlir"
        lower_to_llvm_dialect(str(optimized_mlir), str(lowered_mlir), config)
        temp_files.append(lowered_mlir)
        
        # Step 4: Translate to LLVM IR
        print("Translating to LLVM IR...")
        llvm_ir_file = output_dir / "matmul_chain.ll"
        translate_to_llvm_ir(str(lowered_mlir), str(llvm_ir_file), config)
        
        with open(llvm_ir_file, 'r') as f:
            llvm_ir = f.read()
        print("LLVM IR (first 500 chars):")
        print(llvm_ir[:500] + "..." if len(llvm_ir) > 500 else llvm_ir)
        print()
        
        # Step 5: Compile to object file
        print("Compiling to object file...")
        obj_file = output_dir / "matmul_chain.o"
        compile_to_object(str(llvm_ir_file), str(obj_file), config)
        print(f"Object file generated: {obj_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        # Cleanup temporary files if requested
        if not args.keep_temps:
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()

if __name__ == "__main__":
    main()
