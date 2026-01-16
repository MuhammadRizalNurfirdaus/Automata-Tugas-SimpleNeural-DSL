"""
Command Line Interface untuk SimpleNeural-DSL Compiler
"""

import argparse
import sys
from pathlib import Path
from .compiler import Compiler
from .lexer import Lexer
from .parser import Parser


def cmd_compile(args):
    """Compile DSL file to Python"""
    compiler = Compiler(verbose=args.verbose)
    
    success = compiler.compile_file(
        input_file=args.input,
        output_file=args.output
    )
    
    return 0 if success else 1


def cmd_validate(args):
    """Validate DSL file without compiling"""
    compiler = Compiler(verbose=args.verbose)
    success = compiler.validate_file(args.input)
    return 0 if success else 1


def cmd_tokenize(args):
    """Show tokens for debugging"""
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except FileNotFoundError:
        print(f"❌ Error: File '{args.input}' not found")
        return 1
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return 1
    
    print(f"Tokenizing: {args.input}")
    print("=" * 60)
    
    try:
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        lexer.print_tokens()
        print(f"\nTotal tokens: {len(tokens)}")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_ast(args):
    """Show AST for debugging"""
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except FileNotFoundError:
        print(f"❌ Error: File '{args.input}' not found")
        return 1
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return 1
    
    print(f"Parsing: {args.input}")
    print("=" * 60)
    
    try:
        # Tokenize
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        
        # Parse
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Print AST
        print("\nAbstract Syntax Tree:")
        print("=" * 60)
        parser.print_ast(ast)
        
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_run(args):
    """Compile and run the generated Python code"""
    import subprocess
    import tempfile
    
    # First compile
    compiler = Compiler(verbose=args.verbose)
    
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except FileNotFoundError:
        print(f"❌ Error: File '{args.input}' not found")
        return 1
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return 1
    
    result = compiler.compile_string(source_code, args.input)
    
    if not result['success']:
        print("❌ Compilation failed, cannot run")
        return 1
    
    # Write to temporary file or specified output
    if args.output:
        output_file = args.output
    else:
        output_file = Path(args.input).with_suffix('.py')
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result['code'])
        
        print(f"\n✅ Compiled to: {output_file}")
        print("=" * 60)
        print("Running generated Python code...")
        print("=" * 60)
        
        # Run the generated Python file
        subprocess.run([sys.executable, str(output_file)], check=True)
        
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Runtime error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog='simpleneural',
        description='SimpleNeural-DSL Compiler - Domain Specific Language for ML Configuration',
        epilog='Example: simpleneural compile model.sndsl -o output.py'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='SimpleNeural-DSL 1.0.0'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Compile command
    parser_compile = subparsers.add_parser(
        'compile',
        help='Compile DSL file to Python code'
    )
    parser_compile.add_argument(
        'input',
        help='Input .sndsl file'
    )
    parser_compile.add_argument(
        '-o', '--output',
        help='Output .py file (default: same name as input with .py extension)',
        default=None
    )
    parser_compile.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output (show all compilation stages)'
    )
    parser_compile.set_defaults(func=cmd_compile)
    
    # Validate command
    parser_validate = subparsers.add_parser(
        'validate',
        help='Validate DSL file without generating code'
    )
    parser_validate.add_argument(
        'input',
        help='Input .sndsl file'
    )
    parser_validate.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser_validate.set_defaults(func=cmd_validate)
    
    # Run command
    parser_run = subparsers.add_parser(
        'run',
        help='Compile and run the generated Python code'
    )
    parser_run.add_argument(
        'input',
        help='Input .sndsl file'
    )
    parser_run.add_argument(
        '-o', '--output',
        help='Output .py file (default: same name as input with .py extension)',
        default=None
    )
    parser_run.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser_run.set_defaults(func=cmd_run)
    
    # Tokenize command (debug)
    parser_tokenize = subparsers.add_parser(
        'tokenize',
        help='Show tokens (for debugging)'
    )
    parser_tokenize.add_argument(
        'input',
        help='Input .sndsl file'
    )
    parser_tokenize.set_defaults(func=cmd_tokenize)
    
    # AST command (debug)
    parser_ast = subparsers.add_parser(
        'ast',
        help='Show Abstract Syntax Tree (for debugging)'
    )
    parser_ast.add_argument(
        'input',
        help='Input .sndsl file'
    )
    parser_ast.set_defaults(func=cmd_ast)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show help if no command
    if not args.command:
        parser.print_help()
        return 0
    
    # Execute command
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
