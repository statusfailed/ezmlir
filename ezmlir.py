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

def compile_to_shared_lib(llvm_ir_file, so_file, config):
    """Compile LLVM IR to shared library using clang"""
    cmd = [config.clang, "-shared", "-fPIC", llvm_ir_file, "-o", so_file]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="MLIR compiler using LLVM tools")
    parser.add_argument("input", help="Input MLIR file or '-' to read from stdin")
    parser.add_argument("output", help="Output file name (.o for object file, .so for shared library)")
    parser.add_argument("--suffix", default="", help="Suffix for all LLVM binaries (e.g., '-20' for Ubuntu)")
    parser.add_argument("--mlir-opt", help="Path to mlir-opt binary (overrides --suffix)")
    parser.add_argument("--mlir-translate", help="Path to mlir-translate binary (overrides --suffix)")
    parser.add_argument("--llc", help="Path to llc binary (overrides --suffix)")
    parser.add_argument("--clang", help="Path to clang binary (overrides --suffix)")
    parser.add_argument("--temp-dir", help="Directory for temporary files (if not specified, uses system temp)")
    parser.add_argument("--shared-lib", action="store_true", help="Create shared library (.so) instead of object file (.o)")

    args = parser.parse_args()

    config = MLIRConfig()
    config.mlir_opt = args.mlir_opt or f"mlir-opt{args.suffix}"
    config.mlir_translate = args.mlir_translate or f"mlir-translate{args.suffix}"
    config.llc = args.llc or f"llc{args.suffix}"
    config.clang = args.clang or f"clang{args.suffix}"

    # Read MLIR code from input
    if args.input == '-':
        mlir_code = sys.stdin.read()
    else:
        with open(args.input, 'r') as f:
            mlir_code = f.read()

    # Use specified temp dir or create a temporary one
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
        temp_dir.mkdir(exist_ok=True)
        temp_context = temp_dir
        cleanup_temp = False
    else:
        temp_context = tempfile.TemporaryDirectory()
        cleanup_temp = True

    try:
        if cleanup_temp:
            temp_path = Path(temp_context.name)
        else:
            temp_path = temp_context

        print(f"Using temporary directory: {temp_path}")

        # Get base name from output file for intermediate files
        output_path = Path(args.output)
        base_name = output_path.stem

        original_mlir = temp_path / "original.mlir"
        with open(original_mlir, 'w') as f:
            f.write(mlir_code)

        print("Original MLIR:")
        print(mlir_code)
        print()

        # Step 2: Optimize MLIR
        print("Optimizing MLIR...")
        optimized_mlir = temp_path / "optimized.mlir"
        optimize_mlir(str(original_mlir), str(optimized_mlir), config)

        with open(optimized_mlir, 'r') as f:
            optimized_code = f.read()
        print("After optimization:")
        print(optimized_code)
        print()

        # Step 3: Lower to LLVM dialect
        print("Lowering to LLVM dialect...")
        lowered_mlir = temp_path / "lowered.mlir"
        lower_to_llvm_dialect(str(optimized_mlir), str(lowered_mlir), config)

        # Step 4: Translate to LLVM IR
        print("Translating to LLVM IR...")
        llvm_ir_file = temp_path / f"{base_name}.ll"
        translate_to_llvm_ir(str(lowered_mlir), str(llvm_ir_file), config)

        with open(llvm_ir_file, 'r') as f:
            llvm_ir = f.read()
        print("LLVM IR (first 500 chars):")
        print(llvm_ir[:500] + "..." if len(llvm_ir) > 500 else llvm_ir)
        print()

        # Step 5: Compile to final output
        output_file = Path(args.output)
        if args.shared_lib:
            print("Compiling to shared library...")
            compile_to_shared_lib(str(llvm_ir_file), str(output_file), config)
            print(f"Shared library generated: {output_file}")
        else:
            print("Compiling to object file...")
            compile_to_object(str(llvm_ir_file), str(output_file), config)
            print(f"Object file generated: {output_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        # Cleanup temporary directory if we created it
        if cleanup_temp and hasattr(temp_context, 'cleanup'):
            temp_context.cleanup()

if __name__ == "__main__":
    main()
