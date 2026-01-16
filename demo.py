#!/usr/bin/env python3
"""
Demo Script - SimpleNeural-DSL Compiler
Demonstrasi lengkap fitur compiler
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simpleneural import Compiler


def print_banner(title):
    """Print banner"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_1_minimal():
    """Demo 1: Minimal Example"""
    print_banner("DEMO 1: Minimal Model")
    
    dsl_code = '''
DATASET load "data.csv" TARGET "y"

MODEL "MinimalModel" {
    LAYER DENSE units: 32 activation: "relu"
    LAYER DENSE units: 1 activation: "linear"
    
    OPTIMIZER "adam" lr: 0.01
    TRAIN epochs: 50 batch_size: 32 validation_split: 0.2
}
'''
    
    print("\nDSL Code:")
    print("-" * 70)
    print(dsl_code)
    
    print("\nCompiling...")
    compiler = Compiler(verbose=False)
    result = compiler.compile_string(dsl_code, "demo1.sndsl")
    
    if result['success']:
        print("✅ Compilation successful!")
        print(f"Generated {len(result['code'].splitlines())} lines of Python code")
    else:
        print("❌ Compilation failed!")
        for error in result['errors']:
            print(f"  • {error}")


def demo_2_regression():
    """Demo 2: Regression Model"""
    print_banner("DEMO 2: Regression Model with Dropout")
    
    dsl_code = '''
# House Price Prediction Model
DATASET load "housing_data.csv" TARGET "price"

MODEL "HousePricePredictor" {
    LAYER DENSE units: 128 activation: "relu"
    LAYER DROPOUT rate: 0.3
    LAYER DENSE units: 64 activation: "relu"
    LAYER DROPOUT rate: 0.2
    LAYER DENSE units: 32 activation: "relu"
    LAYER DENSE units: 1 activation: "linear"
    
    OPTIMIZER "adam" lr: 0.001
    TRAIN epochs: 100 batch_size: 32 validation_split: 0.2
}
'''
    
    print("\nDSL Code:")
    print("-" * 70)
    print(dsl_code)
    
    print("\nCompiling with verbose mode...")
    compiler = Compiler(verbose=True)
    result = compiler.compile_string(dsl_code, "demo2.sndsl")
    
    if result['success']:
        print("\n✅ Compilation successful!")


def demo_3_classification():
    """Demo 3: Classification Model"""
    print_banner("DEMO 3: Multi-Class Classification")
    
    dsl_code = '''
# Iris Classification
DATASET load "iris.csv" TARGET "species"

MODEL "IrisClassifier" {
    LAYER DENSE units: 64 activation: "relu"
    LAYER BATCHNORM
    LAYER DENSE units: 32 activation: "relu"
    LAYER DROPOUT rate: 0.2
    LAYER DENSE units: 3 activation: "softmax"
    
    OPTIMIZER "adam" lr: 0.01
    TRAIN epochs: 50 batch_size: 16 validation_split: 0.15
}
'''
    
    print("\nDSL Code:")
    print("-" * 70)
    print(dsl_code)
    
    print("\nCompiling...")
    compiler = Compiler(verbose=False)
    result = compiler.compile_string(dsl_code, "demo3.sndsl")
    
    if result['success']:
        print("✅ Compilation successful!")
        print(f"Generated {len(result['code'].splitlines())} lines of Python code")


def demo_4_lstm():
    """Demo 4: LSTM for Time Series"""
    print_banner("DEMO 4: LSTM Time Series Model")
    
    dsl_code = '''
# Stock Price Prediction with LSTM
DATASET load "stock_prices.csv" TARGET "price"

MODEL "StockPredictor" {
    LAYER LSTM units: 128 return_sequences: true
    LAYER DROPOUT rate: 0.3
    LAYER LSTM units: 64
    LAYER DROPOUT rate: 0.2
    LAYER DENSE units: 32 activation: "relu"
    LAYER DENSE units: 1 activation: "linear"
    
    OPTIMIZER "adam" lr: 0.001
    TRAIN epochs: 100 batch_size: 32 validation_split: 0.2
}
'''
    
    print("\nDSL Code:")
    print("-" * 70)
    print(dsl_code)
    
    print("\nCompiling...")
    compiler = Compiler(verbose=False)
    result = compiler.compile_string(dsl_code, "demo4.sndsl")
    
    if result['success']:
        print("✅ Compilation successful!")
        print(f"Generated {len(result['code'].splitlines())} lines of Python code")
        
        # Show sample of generated code
        print("\nSample of Generated Code (Model Definition):")
        print("-" * 70)
        lines = result['code'].splitlines()
        in_model = False
        count = 0
        for line in lines:
            if "MODEL DEFINITION" in line:
                in_model = True
            if in_model:
                print(line)
                count += 1
                if count > 25:
                    break


def demo_5_error_detection():
    """Demo 5: Error Detection"""
    print_banner("DEMO 5: Error Detection & Validation")
    
    # Error 1: Invalid activation
    print("\n1. Testing Invalid Activation Function:")
    print("-" * 70)
    
    dsl_code = '''
DATASET load "data.csv" TARGET "y"

MODEL "ErrorModel" {
    LAYER DENSE units: 64 activation: "invalid_func"
}
'''
    
    print(dsl_code)
    print("\nCompiling...")
    compiler = Compiler(verbose=False)
    result = compiler.compile_string(dsl_code, "error1.sndsl")
    
    if not result['success']:
        print("✅ Error detected successfully!")
        for error in result['errors']:
            print(f"  ❌ {error}")
    
    # Error 2: Invalid dropout rate
    print("\n2. Testing Invalid Dropout Rate:")
    print("-" * 70)
    
    dsl_code = '''
DATASET load "data.csv" TARGET "y"

MODEL "ErrorModel" {
    LAYER DENSE units: 64 activation: "relu"
    LAYER DROPOUT rate: 1.5
}
'''
    
    print(dsl_code)
    print("\nCompiling...")
    result = compiler.compile_string(dsl_code, "error2.sndsl")
    
    if not result['success']:
        print("✅ Error detected successfully!")
        for error in result['errors']:
            print(f"  ❌ {error}")


def demo_6_code_output():
    """Demo 6: Show Generated Code"""
    print_banner("DEMO 6: Generated Python Code Preview")
    
    dsl_code = '''
DATASET load "data.csv" TARGET "target"

MODEL "DemoModel" {
    LAYER DENSE units: 64 activation: "relu"
    LAYER DENSE units: 32 activation: "relu"
    LAYER DENSE units: 1 activation: "linear"
    
    OPTIMIZER "adam" lr: 0.01
    TRAIN epochs: 50 batch_size: 32 validation_split: 0.2
}
'''
    
    print("\nDSL Code:")
    print("-" * 70)
    print(dsl_code)
    
    print("\nGenerating Python code...")
    compiler = Compiler(verbose=False)
    result = compiler.compile_string(dsl_code, "demo6.sndsl")
    
    if result['success']:
        print("\n✅ Generated Python Code:")
        print("=" * 70)
        print(result['code'])


def main():
    """Main demo function"""
    print("=" * 70)
    print("  SimpleNeural-DSL Compiler - Interactive Demo")
    print("=" * 70)
    print("\nThis demo will showcase the compiler's capabilities")
    print("from simple to complex models, including error detection.")
    
    demos = [
        ("Minimal Model", demo_1_minimal),
        ("Regression Model", demo_2_regression),
        ("Classification Model", demo_3_classification),
        ("LSTM Time Series", demo_4_lstm),
        ("Error Detection", demo_5_error_detection),
        ("Code Output Preview", demo_6_code_output),
    ]
    
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print("  0. Run all demos")
    print("  q. Quit")
    
    while True:
        try:
            choice = input("\nSelect demo (0-6, q to quit): ").strip().lower()
            
            if choice == 'q':
                print("\nThank you for trying SimpleNeural-DSL!")
                break
            elif choice == '0':
                for name, demo_func in demos:
                    demo_func()
                    input("\nPress Enter to continue...")
                print("\n🎉 All demos completed!")
                break
            elif choice.isdigit() and 1 <= int(choice) <= len(demos):
                idx = int(choice) - 1
                demos[idx][1]()
                input("\nPress Enter to continue...")
            else:
                print("Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\n\nDemo interrupted.")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
