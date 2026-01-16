"""
Lexical Analyzer (Tokenizer) untuk SimpleNeural-DSL
Menggunakan Regular Expression dan Finite Automata
"""

import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional


class TokenType(Enum):
    """Token types untuk SimpleNeural-DSL"""
    # Keywords
    KEYWORD_DATASET = auto()
    KEYWORD_MODEL = auto()
    KEYWORD_LAYER = auto()
    KEYWORD_DENSE = auto()
    KEYWORD_CONV2D = auto()
    KEYWORD_DROPOUT = auto()
    KEYWORD_FLATTEN = auto()
    KEYWORD_LSTM = auto()
    KEYWORD_GRU = auto()
    KEYWORD_BATCHNORM = auto()
    KEYWORD_MAXPOOL2D = auto()
    KEYWORD_OPTIMIZER = auto()
    KEYWORD_TRAIN = auto()
    KEYWORD_LOAD = auto()
    KEYWORD_TARGET = auto()
    KEYWORD_UNITS = auto()
    KEYWORD_ACTIVATION = auto()
    KEYWORD_LR = auto()
    KEYWORD_EPOCHS = auto()
    KEYWORD_BATCH_SIZE = auto()
    KEYWORD_VALIDATION_SPLIT = auto()
    KEYWORD_RATE = auto()
    KEYWORD_FILTERS = auto()
    KEYWORD_KERNEL_SIZE = auto()
    KEYWORD_POOL_SIZE = auto()
    KEYWORD_RETURN_SEQUENCES = auto()
    KEYWORD_MOMENTUM = auto()
    
    # Literals
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    TUPLE = auto()
    
    # Identifiers
    IDENTIFIER = auto()
    
    # Punctuation
    LBRACE = auto()
    RBRACE = auto()
    LPAREN = auto()
    RPAREN = auto()
    COLON = auto()
    COMMA = auto()
    
    # Special
    NEWLINE = auto()
    COMMENT = auto()
    EOF = auto()
    UNKNOWN = auto()


@dataclass
class Token:
    """Representasi token"""
    type: TokenType
    value: str
    line: int
    column: int
    
    def __repr__(self):
        return f"Token({self.type.name}, '{self.value}', {self.line}:{self.column})"


class LexerError(Exception):
    """Exception untuk lexical error"""
    def __init__(self, message: str, line: int, column: int):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"Lexical Error at line {line}, column {column}: {message}")


class Lexer:
    """Lexical Analyzer menggunakan Regular Expression"""
    
    # Token patterns - urutan penting! (keyword sebelum identifier)
    TOKEN_PATTERNS = [
        # Keywords - case sensitive
        (r'\bDATASET\b', TokenType.KEYWORD_DATASET),
        (r'\bMODEL\b', TokenType.KEYWORD_MODEL),
        (r'\bLAYER\b', TokenType.KEYWORD_LAYER),
        (r'\bDENSE\b', TokenType.KEYWORD_DENSE),
        (r'\bCONV2D\b', TokenType.KEYWORD_CONV2D),
        (r'\bDROPOUT\b', TokenType.KEYWORD_DROPOUT),
        (r'\bFLATTEN\b', TokenType.KEYWORD_FLATTEN),
        (r'\bLSTM\b', TokenType.KEYWORD_LSTM),
        (r'\bGRU\b', TokenType.KEYWORD_GRU),
        (r'\bBATCHNORM\b', TokenType.KEYWORD_BATCHNORM),
        (r'\bMAXPOOL2D\b', TokenType.KEYWORD_MAXPOOL2D),
        (r'\bOPTIMIZER\b', TokenType.KEYWORD_OPTIMIZER),
        (r'\bTRAIN\b', TokenType.KEYWORD_TRAIN),
        (r'\bload\b', TokenType.KEYWORD_LOAD),
        (r'\bTARGET\b', TokenType.KEYWORD_TARGET),
        (r'\bunits\b', TokenType.KEYWORD_UNITS),
        (r'\bactivation\b', TokenType.KEYWORD_ACTIVATION),
        (r'\blr\b', TokenType.KEYWORD_LR),
        (r'\bepochs\b', TokenType.KEYWORD_EPOCHS),
        (r'\bbatch_size\b', TokenType.KEYWORD_BATCH_SIZE),
        (r'\bvalidation_split\b', TokenType.KEYWORD_VALIDATION_SPLIT),
        (r'\brate\b', TokenType.KEYWORD_RATE),
        (r'\bfilters\b', TokenType.KEYWORD_FILTERS),
        (r'\bkernel_size\b', TokenType.KEYWORD_KERNEL_SIZE),
        (r'\bpool_size\b', TokenType.KEYWORD_POOL_SIZE),
        (r'\breturn_sequences\b', TokenType.KEYWORD_RETURN_SEQUENCES),
        (r'\bmomentum\b', TokenType.KEYWORD_MOMENTUM),
        (r'\btrue\b', TokenType.BOOLEAN),
        (r'\bfalse\b', TokenType.BOOLEAN),
        
        # Literals
        (r'\d+\.\d+', TokenType.FLOAT),
        (r'\d+', TokenType.INTEGER),
        (r'"[^"]*"', TokenType.STRING),
        (r'\(\s*\d+\s*,\s*\d+\s*\)', TokenType.TUPLE),  # (3,3) atau (2,2)
        
        # Identifiers
        (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenType.IDENTIFIER),
        
        # Punctuation
        (r'\{', TokenType.LBRACE),
        (r'\}', TokenType.RBRACE),
        (r'\(', TokenType.LPAREN),
        (r'\)', TokenType.RPAREN),
        (r':', TokenType.COLON),
        (r',', TokenType.COMMA),
        
        # Comments (will be skipped)
        (r'#[^\n]*', TokenType.COMMENT),
        
        # Newline
        (r'\n', TokenType.NEWLINE),
    ]
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        
    def tokenize(self) -> List[Token]:
        """Tokenize source code and return list of tokens"""
        self.tokens = []
        
        while self.position < len(self.source_code):
            # Skip whitespace (except newlines)
            if self.source_code[self.position] in ' \t':
                self.advance()
                continue
                
            # Try to match token
            matched = False
            for pattern, token_type in self.TOKEN_PATTERNS:
                regex = re.compile(pattern)
                match = regex.match(self.source_code, self.position)
                
                if match:
                    value = match.group(0)
                    
                    # Skip comments
                    if token_type == TokenType.COMMENT:
                        self.advance(len(value))
                        matched = True
                        break
                    
                    # Create token
                    token = Token(token_type, value, self.line, self.column)
                    self.tokens.append(token)
                    
                    # Update position
                    if token_type == TokenType.NEWLINE:
                        self.line += 1
                        self.column = 1
                        self.position += 1
                    else:
                        self.advance(len(value))
                    
                    matched = True
                    break
            
            if not matched:
                # Unknown character
                char = self.source_code[self.position]
                raise LexerError(
                    f"Unexpected character: '{char}'",
                    self.line,
                    self.column
                )
        
        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        
        return self.tokens
    
    def advance(self, count: int = 1):
        """Move position forward"""
        for _ in range(count):
            if self.position < len(self.source_code):
                self.position += 1
                self.column += 1
    
    def get_tokens(self) -> List[Token]:
        """Return all tokens"""
        return self.tokens
    
    def print_tokens(self):
        """Print all tokens for debugging"""
        print("=" * 60)
        print("TOKENIZATION RESULT")
        print("=" * 60)
        for token in self.tokens:
            if token.type != TokenType.NEWLINE and token.type != TokenType.EOF:
                print(f"{token.line:3d}:{token.column:3d}  {token.type.name:20s}  '{token.value}'")
        print("=" * 60)


def main():
    """Test lexer"""
    test_code = '''# SimpleNeural DSL Test
DATASET load "data.csv" TARGET "price"

MODEL "TestModel" {
    LAYER DENSE units: 64 activation: "relu"
    LAYER DROPOUT rate: 0.2
    LAYER DENSE units: 1 activation: "linear"
    
    OPTIMIZER "adam" lr: 0.01
    TRAIN epochs: 50 batch_size: 32
}
'''
    
    print("Source Code:")
    print("-" * 60)
    print(test_code)
    print()
    
    try:
        lexer = Lexer(test_code)
        tokens = lexer.tokenize()
        lexer.print_tokens()
        
        print(f"\nTotal tokens: {len(tokens)}")
        
    except LexerError as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
