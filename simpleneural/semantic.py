"""
Semantic Analyzer untuk SimpleNeural-DSL
Melakukan validasi semantik dan type checking
"""

from typing import List, Dict, Any, Optional
from .parser import (
    ProgramNode, DatasetNode, ModelNode, LayerNode,
    OptimizerNode, TrainConfigNode, ASTNode
)


class SemanticError(Exception):
    """Exception for semantic errors"""
    def __init__(self, message: str, line: int = 0, column: int = 0):
        self.message = message
        self.line = line
        self.column = column
        if line > 0:
            super().__init__(f"Semantic Error at line {line}, column {column}: {message}")
        else:
            super().__init__(f"Semantic Error: {message}")


class SymbolTable:
    """Symbol table untuk tracking definitions"""
    def __init__(self):
        self.symbols: Dict[str, Any] = {}
        
    def define(self, name: str, value: Any):
        """Define a symbol"""
        if name in self.symbols:
            raise SemanticError(f"Symbol '{name}' already defined")
        self.symbols[name] = value
        
    def lookup(self, name: str) -> Optional[Any]:
        """Lookup a symbol"""
        return self.symbols.get(name)
    
    def exists(self, name: str) -> bool:
        """Check if symbol exists"""
        return name in self.symbols


class SemanticAnalyzer:
    """Semantic Analyzer dengan type checking dan validation"""
    
    # Valid activation functions
    VALID_ACTIVATIONS = [
        'relu', 'sigmoid', 'tanh', 'softmax', 'linear',
        'selu', 'elu', 'swish', 'gelu'
    ]
    
    # Valid optimizers
    VALID_OPTIMIZERS = [
        'adam', 'sgd', 'rmsprop', 'adagrad', 'adamw', 'nadam'
    ]
    
    # Valid layer types
    VALID_LAYER_TYPES = [
        'DENSE', 'CONV2D', 'DROPOUT', 'FLATTEN',
        'LSTM', 'GRU', 'BATCHNORM', 'MAXPOOL2D'
    ]
    
    # Layer parameter requirements
    LAYER_PARAMS = {
        'DENSE': {
            'required': ['units', 'activation'],
            'optional': []
        },
        'CONV2D': {
            'required': ['filters', 'kernel_size'],
            'optional': ['activation']
        },
        'DROPOUT': {
            'required': ['rate'],
            'optional': []
        },
        'FLATTEN': {
            'required': [],
            'optional': []
        },
        'LSTM': {
            'required': ['units'],
            'optional': ['return_sequences']
        },
        'GRU': {
            'required': ['units'],
            'optional': ['return_sequences']
        },
        'BATCHNORM': {
            'required': [],
            'optional': []
        },
        'MAXPOOL2D': {
            'required': ['pool_size'],
            'optional': []
        }
    }
    
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def analyze(self, ast: ProgramNode) -> bool:
        """Analyze AST and return True if valid"""
        try:
            self.check_program(ast)
            return len(self.errors) == 0
        except SemanticError as e:
            self.errors.append(str(e))
            return False
    
    def check_program(self, program: ProgramNode):
        """Validate program structure"""
        # Check dataset
        if program.dataset:
            self.check_dataset(program.dataset)
        else:
            self.warnings.append("No dataset defined")
        
        # Check models
        if len(program.models) == 0:
            raise SemanticError("Program must have at least one model")
        
        for model in program.models:
            self.check_model(model)
    
    def check_dataset(self, dataset: DatasetNode):
        """Validate dataset configuration"""
        # Check file path
        if not dataset.file_path:
            raise SemanticError(
                "Dataset file path cannot be empty",
                dataset.line,
                dataset.column
            )
        
        # Check file extension
        if not dataset.file_path.endswith('.csv'):
            self.warnings.append(
                f"Dataset file '{dataset.file_path}' is not a .csv file. "
                "Make sure the file format is supported."
            )
        
        # Check target column
        if not dataset.target_column:
            raise SemanticError(
                "Target column cannot be empty",
                dataset.line,
                dataset.column
            )
        
        # Register in symbol table
        self.symbol_table.define('dataset', dataset)
    
    def check_model(self, model: ModelNode):
        """Validate model configuration"""
        # Check model name
        if not model.name:
            raise SemanticError(
                "Model name cannot be empty",
                model.line,
                model.column
            )
        
        # Check for duplicate model names
        if self.symbol_table.exists(f"model_{model.name}"):
            raise SemanticError(
                f"Model '{model.name}' already defined",
                model.line,
                model.column
            )
        
        self.symbol_table.define(f"model_{model.name}", model)
        
        # Check layers
        if len(model.layers) == 0:
            raise SemanticError(
                f"Model '{model.name}' must have at least one layer",
                model.line,
                model.column
            )
        
        for i, layer in enumerate(model.layers):
            self.check_layer(layer, model.name, i)
        
        # Check optimizer
        if model.optimizer:
            self.check_optimizer(model.optimizer, model.name)
        else:
            self.warnings.append(
                f"Model '{model.name}' has no optimizer defined. "
                "Default optimizer will be used."
            )
        
        # Check training config
        if model.train_config:
            self.check_train_config(model.train_config, model.name)
        else:
            self.warnings.append(
                f"Model '{model.name}' has no training configuration. "
                "Default training parameters will be used."
            )
    
    def check_layer(self, layer: LayerNode, model_name: str, layer_index: int):
        """Validate layer configuration"""
        # Check layer type
        if layer.layer_type not in self.VALID_LAYER_TYPES:
            raise SemanticError(
                f"Invalid layer type '{layer.layer_type}' in model '{model_name}'",
                layer.line,
                layer.column
            )
        
        # Get parameter requirements
        param_spec = self.LAYER_PARAMS.get(layer.layer_type, {})
        required_params = param_spec.get('required', [])
        
        # Check required parameters
        for param in required_params:
            if param not in layer.parameters:
                raise SemanticError(
                    f"Layer {layer.layer_type} in model '{model_name}' "
                    f"missing required parameter '{param}'",
                    layer.line,
                    layer.column
                )
        
        # Validate parameter values
        for param_name, param_value in layer.parameters.items():
            self.check_layer_parameter(
                layer.layer_type,
                param_name,
                param_value,
                model_name,
                layer.line,
                layer.column
            )
    
    def check_layer_parameter(
        self,
        layer_type: str,
        param_name: str,
        param_value: Any,
        model_name: str,
        line: int,
        column: int
    ):
        """Validate individual layer parameter"""
        
        # Check units (must be positive integer)
        if param_name == 'units':
            if not isinstance(param_value, int) or param_value <= 0:
                raise SemanticError(
                    f"Parameter 'units' must be a positive integer, got {param_value}",
                    line, column
                )
        
        # Check activation function
        elif param_name == 'activation':
            if param_value not in self.VALID_ACTIVATIONS:
                raise SemanticError(
                    f"Invalid activation function '{param_value}'. "
                    f"Valid options: {', '.join(self.VALID_ACTIVATIONS)}",
                    line, column
                )
        
        # Check dropout rate
        elif param_name == 'rate':
            if not isinstance(param_value, (int, float)):
                raise SemanticError(
                    f"Parameter 'rate' must be a number, got {type(param_value).__name__}",
                    line, column
                )
            if not 0.0 <= param_value <= 1.0:
                raise SemanticError(
                    f"Dropout rate must be between 0.0 and 1.0, got {param_value}",
                    line, column
                )
        
        # Check filters (for CONV2D)
        elif param_name == 'filters':
            if not isinstance(param_value, int) or param_value <= 0:
                raise SemanticError(
                    f"Parameter 'filters' must be a positive integer, got {param_value}",
                    line, column
                )
        
        # Check kernel_size and pool_size (tuples)
        elif param_name in ['kernel_size', 'pool_size']:
            # Value should be string like "(3,3)"
            if not isinstance(param_value, str):
                raise SemanticError(
                    f"Parameter '{param_name}' must be a tuple like (3,3)",
                    line, column
                )
        
        # Check return_sequences (boolean)
        elif param_name == 'return_sequences':
            if not isinstance(param_value, bool):
                raise SemanticError(
                    f"Parameter 'return_sequences' must be boolean (true/false)",
                    line, column
                )
    
    def check_optimizer(self, optimizer: OptimizerNode, model_name: str):
        """Validate optimizer configuration"""
        # Check optimizer type
        if optimizer.optimizer_type not in self.VALID_OPTIMIZERS:
            raise SemanticError(
                f"Invalid optimizer '{optimizer.optimizer_type}'. "
                f"Valid options: {', '.join(self.VALID_OPTIMIZERS)}",
                optimizer.line,
                optimizer.column
            )
        
        # Check learning rate
        if 'lr' in optimizer.parameters:
            lr = optimizer.parameters['lr']
            if not isinstance(lr, (int, float)):
                raise SemanticError(
                    f"Learning rate must be a number, got {type(lr).__name__}",
                    optimizer.line,
                    optimizer.column
                )
            if not 0.0001 <= lr <= 1.0:
                self.warnings.append(
                    f"Learning rate {lr} is outside typical range [0.0001, 1.0]"
                )
        
        # Check momentum (for SGD)
        if 'momentum' in optimizer.parameters:
            momentum = optimizer.parameters['momentum']
            if not isinstance(momentum, (int, float)):
                raise SemanticError(
                    f"Momentum must be a number, got {type(momentum).__name__}",
                    optimizer.line,
                    optimizer.column
                )
            if not 0.0 <= momentum <= 1.0:
                raise SemanticError(
                    f"Momentum must be between 0.0 and 1.0, got {momentum}",
                    optimizer.line,
                    optimizer.column
                )
    
    def check_train_config(self, train_config: TrainConfigNode, model_name: str):
        """Validate training configuration"""
        # Check epochs
        if 'epochs' in train_config.parameters:
            epochs = train_config.parameters['epochs']
            if not isinstance(epochs, int) or epochs <= 0:
                raise SemanticError(
                    f"Epochs must be a positive integer, got {epochs}",
                    train_config.line,
                    train_config.column
                )
        
        # Check batch_size
        if 'batch_size' in train_config.parameters:
            batch_size = train_config.parameters['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise SemanticError(
                    f"Batch size must be a positive integer, got {batch_size}",
                    train_config.line,
                    train_config.column
                )
        
        # Check validation_split
        if 'validation_split' in train_config.parameters:
            val_split = train_config.parameters['validation_split']
            if not isinstance(val_split, (int, float)):
                raise SemanticError(
                    f"Validation split must be a number, got {type(val_split).__name__}",
                    train_config.line,
                    train_config.column
                )
            if not 0.0 < val_split < 1.0:
                raise SemanticError(
                    f"Validation split must be between 0.0 and 1.0, got {val_split}",
                    train_config.line,
                    train_config.column
                )
    
    def get_errors(self) -> List[str]:
        """Get all errors"""
        return self.errors
    
    def get_warnings(self) -> List[str]:
        """Get all warnings"""
        return self.warnings
    
    def print_report(self):
        """Print analysis report"""
        print("=" * 60)
        print("SEMANTIC ANALYSIS REPORT")
        print("=" * 60)
        
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ No errors or warnings found!")
        
        print("=" * 60)


def main():
    """Test semantic analyzer"""
    from .lexer import Lexer
    from .parser import Parser
    
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
        
        # Parse
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Semantic analysis
        print("Analyzing semantics...")
        analyzer = SemanticAnalyzer()
        is_valid = analyzer.analyze(ast)
        
        analyzer.print_report()
        
        if is_valid:
            print("\n✅ Program is semantically valid!")
        else:
            print("\n❌ Program has semantic errors!")
        
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
