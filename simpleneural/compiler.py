"""
Main Compiler untuk SimpleNeural-DSL
Menggabungkan semua komponen: Lexer, Parser, Semantic Analyzer, dan Code Generator
"""

import sys
from typing import Optional
from pathlib import Path

from .lexer import Lexer, LexerError
from .parser import Parser, ParseError
from .semantic import SemanticAnalyzer, SemanticError
from .codegen import CodeGenerator


class CompilerError(Exception):
    """General compiler error"""
    pass


class Compiler:
    """Main compiler class"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.source_file: Optional[str] = None
        self.source_code: Optional[str] = None
        
    def compile_file(self, input_file: str, output_file: Optional[str] = None) -> dict:
        """
        Compile DSL file to Python code
        
        Args:
            input_file: Path to .sndsl file
            output_file: Path to output .py file (optional)
            
        Returns:
            Dictionary with 'success', 'code', 'ast', 'errors', and 'output_file' keys
        """
        self.source_file = input_file
        
        # Read source file
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                self.source_code = f.read()
        except FileNotFoundError:
            return {
                'success': False,
                'errors': [f"File '{input_file}' not found"],
                'code': None,
                'ast': None,
                'output_file': None
            }
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Error reading file: {e}"],
                'code': None,
                'ast': None,
                'output_file': None
            }
        
        # Compile
        result = self.compile_string(self.source_code, input_file)
        
        if result['success'] and output_file:
            # Write output
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result['code'])
                result['output_file'] = output_file
            except Exception as e:
                result['success'] = False
                result['errors'].append(f"Error writing output file: {e}")
        
        return result
    
    def compile_string(self, source_code: str, source_name: str = "<string>") -> dict:
        """
        Compile DSL source code string
        
        Args:
            source_code: DSL source code
            source_name: Name for error messages
            
        Returns:
            Dictionary with 'success', 'code', and 'errors' keys
        """
        result = {
            'success': False,
            'code': None,
            'ast': None,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Stage 1: Lexical Analysis
            if self.verbose:
                print("\n" + "=" * 60)
                print("STAGE 1: Lexical Analysis")
                print("=" * 60)
            
            lexer = Lexer(source_code)
            tokens = lexer.tokenize()
            
            if self.verbose:
                print(f"✓ Tokenization complete: {len(tokens)} tokens")
                lexer.print_tokens()
            
            # Stage 2: Syntax Analysis (Parsing)
            if self.verbose:
                print("\n" + "=" * 60)
                print("STAGE 2: Syntax Analysis (Parsing)")
                print("=" * 60)
            
            parser = Parser(tokens)
            ast = parser.parse()
            
            if self.verbose:
                print("✓ Parsing complete")
                print("\nAbstract Syntax Tree:")
                parser.print_ast(ast)
            
            # Stage 3: Semantic Analysis
            if self.verbose:
                print("\n" + "=" * 60)
                print("STAGE 3: Semantic Analysis")
                print("=" * 60)
            
            analyzer = SemanticAnalyzer()
            is_valid = analyzer.analyze(ast)
            
            # Get warnings
            result['warnings'] = analyzer.get_warnings()
            
            if not is_valid:
                result['errors'] = analyzer.get_errors()
                if self.verbose:
                    analyzer.print_report()
                else:
                    print("❌ Semantic errors found:")
                    for error in result['errors']:
                        print(f"  • {error}")
                return result
            
            if self.verbose:
                print("✓ Semantic analysis complete")
                if result['warnings']:
                    print("\n⚠️  Warnings:")
                    for warning in result['warnings']:
                        print(f"  • {warning}")
            
            # Stage 4: Code Generation
            if self.verbose:
                print("\n" + "=" * 60)
                print("STAGE 4: Code Generation")
                print("=" * 60)
            
            generator = CodeGenerator()
            python_code = generator.generate(ast, source_name)
            
            if self.verbose:
                print("✓ Code generation complete")
            
            result['success'] = True
            result['code'] = python_code
            result['ast'] = ast
            
            return result
            
        except LexerError as e:
            error_msg = str(e)
            result['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return result
            
        except ParseError as e:
            error_msg = str(e)
            result['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return result
            
        except SemanticError as e:
            error_msg = str(e)
            result['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            result['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return result
    
    def validate_file(self, input_file: str) -> bool:
        """
        Validate DSL file without generating code
        
        Args:
            input_file: Path to .sndsl file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except FileNotFoundError:
            print(f"❌ Error: File '{input_file}' not found")
            return False
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return False
        
        print(f"Validating: {input_file}")
        print("-" * 60)
        
        try:
            # Lexical analysis
            lexer = Lexer(source_code)
            tokens = lexer.tokenize()
            print(f"✓ Lexical analysis passed ({len(tokens)} tokens)")
            
            # Syntax analysis
            parser = Parser(tokens)
            ast = parser.parse()
            print(f"✓ Syntax analysis passed")
            
            # Semantic analysis
            analyzer = SemanticAnalyzer()
            is_valid = analyzer.analyze(ast)
            
            if is_valid:
                print(f"✓ Semantic analysis passed")
                
                warnings = analyzer.get_warnings()
                if warnings:
                    print("\n⚠️  Warnings:")
                    for warning in warnings:
                        print(f"  • {warning}")
                
                print("\n✅ File is valid!")
                return True
            else:
                print(f"❌ Semantic analysis failed")
                errors = analyzer.get_errors()
                for error in errors:
                    print(f"  • {error}")
                return False
                
        except (LexerError, ParseError, SemanticError) as e:
            print(f"❌ {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False


def main():
    """Test compiler"""
    test_code = '''
# SimpleNeural DSL - Test
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
    
    print("SimpleNeural-DSL Compiler Test")
    print("=" * 60)
    
    compiler = Compiler(verbose=True)
    result = compiler.compile_string(test_code, "test.sndsl")
    
    if result['success']:
        print("\n" + "=" * 60)
        print("GENERATED PYTHON CODE")
        print("=" * 60)
        print(result['code'])
    else:
        print("\n❌ Compilation failed!")
        for error in result['errors']:
            print(f"  • {error}")


if __name__ == "__main__":
    main()
