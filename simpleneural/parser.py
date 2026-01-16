"""
Parser dan Abstract Syntax Tree (AST) untuk SimpleNeural-DSL
Menggunakan Recursive Descent Parsing (LL(1))
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from enum import Enum
from .lexer import Token, TokenType, Lexer


class ASTNodeType(Enum):
    """Types of AST nodes"""
    PROGRAM = "program"
    DATASET = "dataset"
    MODEL = "model"
    LAYER = "layer"
    OPTIMIZER = "optimizer"
    TRAIN_CONFIG = "train_config"
    PARAMETER = "parameter"


@dataclass
class ASTNode:
    """Base class for AST nodes"""
    node_type: ASTNodeType
    line: int = 0
    column: int = 0


@dataclass
class ParameterNode(ASTNode):
    """Parameter node: name: value"""
    name: str = ""
    value: Any = None
    
    def __post_init__(self):
        if self.node_type != ASTNodeType.PARAMETER:
            self.node_type = ASTNodeType.PARAMETER


@dataclass
class LayerNode(ASTNode):
    """Layer node"""
    layer_type: str = ""  # DENSE, CONV2D, DROPOUT, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.node_type != ASTNodeType.LAYER:
            self.node_type = ASTNodeType.LAYER


@dataclass
class OptimizerNode(ASTNode):
    """Optimizer configuration node"""
    optimizer_type: str = ""  # adam, sgd, rmsprop
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.node_type != ASTNodeType.OPTIMIZER:
            self.node_type = ASTNodeType.OPTIMIZER


@dataclass
class TrainConfigNode(ASTNode):
    """Training configuration node"""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.node_type != ASTNodeType.TRAIN_CONFIG:
            self.node_type = ASTNodeType.TRAIN_CONFIG


@dataclass
class ModelNode(ASTNode):
    """Model definition node"""
    name: str = ""
    layers: List[LayerNode] = field(default_factory=list)
    optimizer: Optional[OptimizerNode] = None
    train_config: Optional[TrainConfigNode] = None
    
    def __post_init__(self):
        if self.node_type != ASTNodeType.MODEL:
            self.node_type = ASTNodeType.MODEL


@dataclass
class DatasetNode(ASTNode):
    """Dataset configuration node"""
    file_path: str = ""
    target_column: str = ""
    
    def __post_init__(self):
        if self.node_type != ASTNodeType.DATASET:
            self.node_type = ASTNodeType.DATASET


@dataclass
class ProgramNode(ASTNode):
    """Root node of AST"""
    dataset: Optional[DatasetNode] = None
    models: List[ModelNode] = field(default_factory=list)
    
    def __post_init__(self):
        if self.node_type != ASTNodeType.PROGRAM:
            self.node_type = ASTNodeType.PROGRAM


class ParseError(Exception):
    """Exception for parsing errors"""
    def __init__(self, message: str, line: int, column: int):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"Parse Error at line {line}, column {column}: {message}")


class Parser:
    """Recursive Descent Parser for SimpleNeural-DSL"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
        self.current_token = self.tokens[0] if tokens else None
        
    def parse(self) -> ProgramNode:
        """Parse tokens and return AST"""
        return self.parse_program()
    
    def current(self) -> Token:
        """Get current token"""
        return self.current_token
    
    def peek(self, offset: int = 1) -> Optional[Token]:
        """Peek ahead at tokens"""
        pos = self.position + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None
    
    def advance(self):
        """Move to next token"""
        if self.position < len(self.tokens) - 1:
            self.position += 1
            self.current_token = self.tokens[self.position]
    
    def expect(self, token_type: TokenType) -> Token:
        """Expect a specific token type"""
        if self.current().type != token_type:
            raise ParseError(
                f"Expected {token_type.name}, got {self.current().type.name}",
                self.current().line,
                self.current().column
            )
        token = self.current()
        self.advance()
        return token
    
    def skip_newlines(self):
        """Skip all newline tokens"""
        while self.current().type == TokenType.NEWLINE:
            self.advance()
    
    def parse_program(self) -> ProgramNode:
        """
        Program → Dataset? Model+
        """
        program = ProgramNode(node_type=ASTNodeType.PROGRAM)
        
        self.skip_newlines()
        
        # Parse DATASET (optional)
        if self.current().type == TokenType.KEYWORD_DATASET:
            program.dataset = self.parse_dataset()
            self.skip_newlines()
        
        # Parse MODEL (at least one)
        while self.current().type == TokenType.KEYWORD_MODEL:
            model = self.parse_model()
            program.models.append(model)
            self.skip_newlines()
        
        # Expect EOF
        if self.current().type != TokenType.EOF:
            raise ParseError(
                f"Unexpected token: {self.current().value}",
                self.current().line,
                self.current().column
            )
        
        return program
    
    def parse_dataset(self) -> DatasetNode:
        """
        Dataset → DATASET load STRING TARGET STRING
        """
        token = self.expect(TokenType.KEYWORD_DATASET)
        dataset = DatasetNode(
            node_type=ASTNodeType.DATASET,
            line=token.line,
            column=token.column
        )
        
        self.expect(TokenType.KEYWORD_LOAD)
        
        file_token = self.expect(TokenType.STRING)
        dataset.file_path = file_token.value.strip('"')
        
        self.expect(TokenType.KEYWORD_TARGET)
        
        target_token = self.expect(TokenType.STRING)
        dataset.target_column = target_token.value.strip('"')
        
        return dataset
    
    def parse_model(self) -> ModelNode:
        """
        Model → MODEL STRING { Layer+ Optimizer? TrainConfig? }
        """
        token = self.expect(TokenType.KEYWORD_MODEL)
        model = ModelNode(
            node_type=ASTNodeType.MODEL,
            line=token.line,
            column=token.column
        )
        
        # Get model name
        name_token = self.expect(TokenType.STRING)
        model.name = name_token.value.strip('"')
        
        self.skip_newlines()
        self.expect(TokenType.LBRACE)
        self.skip_newlines()
        
        # Parse body: layers, optimizer, train config
        while self.current().type != TokenType.RBRACE:
            if self.current().type == TokenType.KEYWORD_LAYER:
                layer = self.parse_layer()
                model.layers.append(layer)
            elif self.current().type == TokenType.KEYWORD_OPTIMIZER:
                model.optimizer = self.parse_optimizer()
            elif self.current().type == TokenType.KEYWORD_TRAIN:
                model.train_config = self.parse_train_config()
            else:
                raise ParseError(
                    f"Unexpected token in model body: {self.current().value}",
                    self.current().line,
                    self.current().column
                )
            self.skip_newlines()
        
        self.expect(TokenType.RBRACE)
        
        return model
    
    def parse_layer(self) -> LayerNode:
        """
        Layer → LAYER LayerType Parameters*
        LayerType → DENSE | CONV2D | DROPOUT | FLATTEN | LSTM | GRU | BATCHNORM | MAXPOOL2D
        """
        token = self.expect(TokenType.KEYWORD_LAYER)
        layer = LayerNode(
            node_type=ASTNodeType.LAYER,
            line=token.line,
            column=token.column
        )
        
        # Get layer type
        layer_type_token = self.current()
        if layer_type_token.type in [
            TokenType.KEYWORD_DENSE,
            TokenType.KEYWORD_CONV2D,
            TokenType.KEYWORD_DROPOUT,
            TokenType.KEYWORD_FLATTEN,
            TokenType.KEYWORD_LSTM,
            TokenType.KEYWORD_GRU,
            TokenType.KEYWORD_BATCHNORM,
            TokenType.KEYWORD_MAXPOOL2D,
        ]:
            layer.layer_type = layer_type_token.value
            self.advance()
        else:
            raise ParseError(
                f"Expected layer type, got {layer_type_token.value}",
                layer_type_token.line,
                layer_type_token.column
            )
        
        # Parse parameters
        while self.current().type not in [TokenType.NEWLINE, TokenType.RBRACE, TokenType.EOF]:
            param_name, param_value = self.parse_parameter()
            layer.parameters[param_name] = param_value
        
        return layer
    
    def parse_optimizer(self) -> OptimizerNode:
        """
        Optimizer → OPTIMIZER STRING Parameters*
        """
        token = self.expect(TokenType.KEYWORD_OPTIMIZER)
        optimizer = OptimizerNode(
            node_type=ASTNodeType.OPTIMIZER,
            line=token.line,
            column=token.column
        )
        
        # Get optimizer type
        type_token = self.expect(TokenType.STRING)
        optimizer.optimizer_type = type_token.value.strip('"')
        
        # Parse parameters
        while self.current().type not in [TokenType.NEWLINE, TokenType.RBRACE, TokenType.EOF]:
            param_name, param_value = self.parse_parameter()
            optimizer.parameters[param_name] = param_value
        
        return optimizer
    
    def parse_train_config(self) -> TrainConfigNode:
        """
        TrainConfig → TRAIN Parameters+
        """
        token = self.expect(TokenType.KEYWORD_TRAIN)
        train_config = TrainConfigNode(
            node_type=ASTNodeType.TRAIN_CONFIG,
            line=token.line,
            column=token.column
        )
        
        # Parse parameters
        while self.current().type not in [TokenType.NEWLINE, TokenType.RBRACE, TokenType.EOF]:
            param_name, param_value = self.parse_parameter()
            train_config.parameters[param_name] = param_value
        
        return train_config
    
    def parse_parameter(self) -> tuple[str, Any]:
        """
        Parameter → IDENTIFIER : Value
        Value → STRING | INTEGER | FLOAT | BOOLEAN | TUPLE
        """
        # Get parameter name
        name_token = self.current()
        if name_token.type not in [
            TokenType.KEYWORD_UNITS,
            TokenType.KEYWORD_ACTIVATION,
            TokenType.KEYWORD_LR,
            TokenType.KEYWORD_EPOCHS,
            TokenType.KEYWORD_BATCH_SIZE,
            TokenType.KEYWORD_VALIDATION_SPLIT,
            TokenType.KEYWORD_RATE,
            TokenType.KEYWORD_FILTERS,
            TokenType.KEYWORD_KERNEL_SIZE,
            TokenType.KEYWORD_POOL_SIZE,
            TokenType.KEYWORD_RETURN_SEQUENCES,
            TokenType.KEYWORD_MOMENTUM,
        ]:
            raise ParseError(
                f"Expected parameter name, got {name_token.value}",
                name_token.line,
                name_token.column
            )
        
        param_name = name_token.value
        self.advance()
        
        # Expect colon
        self.expect(TokenType.COLON)
        
        # Get parameter value
        value_token = self.current()
        
        if value_token.type == TokenType.STRING:
            param_value = value_token.value.strip('"')
        elif value_token.type == TokenType.INTEGER:
            param_value = int(value_token.value)
        elif value_token.type == TokenType.FLOAT:
            param_value = float(value_token.value)
        elif value_token.type == TokenType.BOOLEAN:
            param_value = value_token.value == "true"
        elif value_token.type == TokenType.TUPLE:
            # Parse tuple like (3,3)
            param_value = value_token.value
        else:
            raise ParseError(
                f"Expected value, got {value_token.value}",
                value_token.line,
                value_token.column
            )
        
        self.advance()
        
        return param_name, param_value
    
    def print_ast(self, node: ASTNode, indent: int = 0):
        """Print AST for debugging"""
        prefix = "  " * indent
        
        if isinstance(node, ProgramNode):
            print(f"{prefix}Program:")
            if node.dataset:
                self.print_ast(node.dataset, indent + 1)
            for model in node.models:
                self.print_ast(model, indent + 1)
                
        elif isinstance(node, DatasetNode):
            print(f"{prefix}Dataset:")
            print(f"{prefix}  file: {node.file_path}")
            print(f"{prefix}  target: {node.target_column}")
            
        elif isinstance(node, ModelNode):
            print(f"{prefix}Model: {node.name}")
            for layer in node.layers:
                self.print_ast(layer, indent + 1)
            if node.optimizer:
                self.print_ast(node.optimizer, indent + 1)
            if node.train_config:
                self.print_ast(node.train_config, indent + 1)
                
        elif isinstance(node, LayerNode):
            print(f"{prefix}Layer: {node.layer_type}")
            for name, value in node.parameters.items():
                print(f"{prefix}  {name}: {value}")
                
        elif isinstance(node, OptimizerNode):
            print(f"{prefix}Optimizer: {node.optimizer_type}")
            for name, value in node.parameters.items():
                print(f"{prefix}  {name}: {value}")
                
        elif isinstance(node, TrainConfigNode):
            print(f"{prefix}TrainConfig:")
            for name, value in node.parameters.items():
                print(f"{prefix}  {name}: {value}")


def main():
    """Test parser"""
    from .lexer import Lexer
    
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
    
    print("Source Code:")
    print("-" * 60)
    print(test_code)
    print()
    
    try:
        # Tokenize
        lexer = Lexer(test_code)
        tokens = lexer.tokenize()
        
        print("Parsing...")
        print("=" * 60)
        
        # Parse
        parser = Parser(tokens)
        ast = parser.parse()
        
        print("\nAST:")
        print("=" * 60)
        parser.print_ast(ast)
        
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
