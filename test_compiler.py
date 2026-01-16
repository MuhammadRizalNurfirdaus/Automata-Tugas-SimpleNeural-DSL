"""
Test script untuk SimpleNeural-DSL Compiler
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simpleneural.lexer import Lexer
from simpleneural.parser import Parser
from simpleneural.semantic import SemanticAnalyzer
from simpleneural.codegen import CodeGenerator
from simpleneural.compiler import Compiler


def test_lexer():
    """Test Lexer"""
    print("\n" + "=" * 60)
    print("TEST 1: LEXER (Tokenization)")
    print("=" * 60)
    
    test_code = '''
DATASET load "data.csv" TARGET "price"

MODEL "TestModel" {
    LAYER DENSE units: 64 activation: "relu"
    LAYER DROPOUT rate: 0.2
}
'''
    
    try:
        lexer = Lexer(test_code)
        tokens = lexer.tokenize()
        print(f"✅ Lexer test passed: {len(tokens)} tokens")
        lexer.print_tokens()
        return True
    except Exception as e:
        print(f"❌ Lexer test failed: {e}")
        return False


def test_parser():
    """Test Parser"""
    print("\n" + "=" * 60)
    print("TEST 2: PARSER (Syntax Analysis)")
    print("=" * 60)
    
    test_code = '''
DATASET load "data.csv" TARGET "price"

MODEL "TestModel" {
    LAYER DENSE units: 64 activation: "relu"
    LAYER DROPOUT rate: 0.2
    OPTIMIZER "adam" lr: 0.01
    TRAIN epochs: 50 batch_size: 32
}
'''
    
    try:
        lexer = Lexer(test_code)
        tokens = lexer.tokenize()
        
        parser = Parser(tokens)
        ast = parser.parse()
        
        print("✅ Parser test passed")
        print("\nAST Structure:")
        parser.print_ast(ast)
        return True
    except Exception as e:
        print(f"❌ Parser test failed: {e}")
        return False


def test_semantic():
    """Test Semantic Analyzer"""
    print("\n" + "=" * 60)
    print("TEST 3: SEMANTIC ANALYZER (Validation)")
    print("=" * 60)
    
    test_code = '''
DATASET load "data.csv" TARGET "price"

MODEL "TestModel" {
    LAYER DENSE units: 64 activation: "relu"
    LAYER DROPOUT rate: 0.2
    LAYER DENSE units: 1 activation: "linear"
    OPTIMIZER "adam" lr: 0.01
    TRAIN epochs: 50 batch_size: 32 validation_split: 0.2
}
'''
    
    try:
        lexer = Lexer(test_code)
        tokens = lexer.tokenize()
        
        parser = Parser(tokens)
        ast = parser.parse()
        
        analyzer = SemanticAnalyzer()
        is_valid = analyzer.analyze(ast)
        
        if is_valid:
            print("✅ Semantic analysis test passed")
            analyzer.print_report()
            return True
        else:
            print("❌ Semantic analysis test failed")
            analyzer.print_report()
            return False
    except Exception as e:
        print(f"❌ Semantic analyzer test failed: {e}")
        return False


def test_codegen():
    """Test Code Generator"""
    print("\n" + "=" * 60)
    print("TEST 4: CODE GENERATOR (Python Output)")
    print("=" * 60)
    
    test_code = '''
DATASET load "data.csv" TARGET "price"

MODEL "TestModel" {
    LAYER DENSE units: 64 activation: "relu"
    LAYER DROPOUT rate: 0.2
    LAYER DENSE units: 32 activation: "relu"
    LAYER DENSE units: 1 activation: "linear"
    OPTIMIZER "adam" lr: 0.01
    TRAIN epochs: 50 batch_size: 32 validation_split: 0.2
}
'''
    
    try:
        lexer = Lexer(test_code)
        tokens = lexer.tokenize()
        
        parser = Parser(tokens)
        ast = parser.parse()
        
        generator = CodeGenerator()
        python_code = generator.generate(ast, "test.sndsl")
        
        print("✅ Code generation test passed")
        print(f"\nGenerated {len(python_code.splitlines())} lines of Python code")
        print("\nFirst 20 lines:")
        print("-" * 60)
        for i, line in enumerate(python_code.splitlines()[:20], 1):
            print(f"{i:3d}: {line}")
        print("...")
        return True
    except Exception as e:
        print(f"❌ Code generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compiler():
    """Test Full Compiler"""
    print("\n" + "=" * 60)
    print("TEST 5: FULL COMPILER (End-to-End)")
    print("=" * 60)
    
    test_code = '''
# SimpleNeural DSL Test
DATASET load "data.csv" TARGET "price"

MODEL "FullTest" {
    LAYER DENSE units: 128 activation: "relu"
    LAYER DROPOUT rate: 0.3
    LAYER DENSE units: 64 activation: "relu"
    LAYER DENSE units: 1 activation: "linear"
    OPTIMIZER "adam" lr: 0.001
    TRAIN epochs: 100 batch_size: 32 validation_split: 0.2
}
'''
    
    try:
        compiler = Compiler(verbose=False)
        result = compiler.compile_string(test_code, "test.sndsl")
        
        if result['success']:
            print("✅ Full compiler test passed")
            print(f"\nGenerated {len(result['code'].splitlines())} lines of code")
            
            if result['warnings']:
                print("\nWarnings:")
                for warning in result['warnings']:
                    print(f"  ⚠️  {warning}")
            
            return True
        else:
            print("❌ Full compiler test failed")
            for error in result['errors']:
                print(f"  ❌ {error}")
            return False
    except Exception as e:
        print(f"❌ Compiler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_detection():
    """Test Error Detection"""
    print("\n" + "=" * 60)
    print("TEST 6: ERROR DETECTION")
    print("=" * 60)
    
    # Test semantic error: invalid activation
    test_code = '''
DATASET load "data.csv" TARGET "price"

MODEL "ErrorTest" {
    LAYER DENSE units: 64 activation: "invalid_activation"
}
'''
    
    try:
        compiler = Compiler(verbose=False)
        result = compiler.compile_string(test_code, "error_test.sndsl")
        
        if not result['success']:
            print("✅ Error detection test passed")
            print("Detected errors:")
            for error in result['errors']:
                print(f"  ❌ {error}")
            return True
        else:
            print("❌ Error detection test failed: Should have detected error")
            return False
    except Exception as e:
        print(f"✅ Error detection test passed: {type(e).__name__}")
        return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("SimpleNeural-DSL Compiler - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Lexer", test_lexer),
        ("Parser", test_parser),
        ("Semantic Analyzer", test_semantic),
        ("Code Generator", test_codegen),
        ("Full Compiler", test_compiler),
        ("Error Detection", test_error_detection),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
