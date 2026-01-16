# SimpleNeural-DSL Project Structure

## 📂 Struktur Direktori

```
automata/
├── README.md                    # Dokumentasi utama
├── QUICKSTART.md               # Panduan cepat
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
├── setup.py                    # Package setup
├── requirements.txt            # Dependencies
├── test_compiler.py            # Test suite
│
├── simpleneural/               # 🔧 Package Utama
│   ├── __init__.py            # Package initialization
│   ├── __main__.py            # Entry point (python -m)
│   ├── lexer.py               # Lexical Analyzer (650+ lines)
│   ├── parser.py              # Syntax Parser (500+ lines)
│   ├── semantic.py            # Semantic Analyzer (400+ lines)
│   ├── codegen.py             # Code Generator (500+ lines)
│   ├── compiler.py            # Main Compiler (250+ lines)
│   └── cli.py                 # CLI Interface (250+ lines)
│
├── examples/                   # 📝 Contoh File DSL
│   ├── minimal.sndsl          # Contoh minimal
│   ├── housing_regression.sndsl    # Prediksi harga rumah
│   ├── iris_classification.sndsl   # Klasifikasi Iris
│   ├── deep_network.sndsl     # Deep neural network
│   ├── lstm_timeseries.sndsl  # Time series LSTM
│   └── error_test.sndsl       # Test error detection
│
└── docs/                       # 📚 Dokumentasi Lengkap
    ├── README.md              # Index dokumentasi
    ├── 01-pendahuluan.md      # Latar belakang & tujuan
    ├── 02-use-case.md         # Use case analysis
    ├── 03-arsitektur.md       # Arsitektur sistem
    ├── 04-grammar-token.md    # Spesifikasi grammar
    ├── 05-implementasi.md     # Detail implementasi
    └── 06-testing-examples.md # Testing & examples
```

## 🎯 Komponen Utama

### 1. Lexer (lexer.py)
- **Fungsi**: Tokenization - mengubah source code menjadi token stream
- **Teknik**: Finite Automata berbasis Regular Expression
- **Token Types**: 30+ jenis token (keywords, literals, punctuation)
- **Error Handling**: LexerError dengan line & column information

**Fitur:**
- Pattern matching menggunakan compiled regex
- Support untuk comments (#)
- Whitespace handling
- String, integer, float, boolean literals
- Tuple literals untuk kernel_size dan pool_size

### 2. Parser (parser.py)
- **Fungsi**: Syntax Analysis - membuat Abstract Syntax Tree (AST)
- **Teknik**: Recursive Descent Parsing (LL(1))
- **AST Nodes**: 7 jenis node (Program, Dataset, Model, Layer, dll)
- **Error Handling**: ParseError dengan informasi posisi

**Fitur:**
- Context-Free Grammar implementation
- Hierarchical AST structure
- Parameter parsing dan validation
- Newline handling
- Pretty-print AST untuk debugging

### 3. Semantic Analyzer (semantic.py)
- **Fungsi**: Semantic validation & type checking
- **Teknik**: Symbol Table, Type System
- **Validasi**: Parameter ranges, types, business rules
- **Error & Warnings**: Detailed error messages dengan suggestions

**Fitur:**
- Valid activation function checking
- Valid optimizer checking
- Parameter range validation (learning rate, dropout rate, etc.)
- Required parameter checking
- Model structure validation
- Symbol table untuk duplicate checking

### 4. Code Generator (codegen.py)
- **Fungsi**: Generate Python code dari validated AST
- **Teknik**: Template-based code generation
- **Output**: Clean, PEP8-compliant Python code
- **Framework**: TensorFlow 2.x / Keras

**Fitur:**
- Modular code generation (imports, data loading, model, training)
- Support untuk semua layer types
- Optimizer configuration
- Training dengan callbacks (EarlyStopping, ReduceLROnPlateau)
- Evaluation metrics (MSE, MAE, R²)
- Model saving

### 5. Compiler (compiler.py)
- **Fungsi**: Orchestrate semua komponen
- **Pipeline**: Lexer → Parser → Semantic → CodeGen
- **Error Handling**: Comprehensive error reporting
- **Modes**: Compile, validate, compile-and-run

**Fitur:**
- Multi-stage compilation
- Verbose mode untuk debugging
- Error collection dan reporting
- Warning system
- File I/O handling

### 6. CLI (cli.py)
- **Fungsi**: Command-line interface
- **Commands**: compile, validate, run, tokenize, ast
- **Features**: Argparse-based, help messages, exit codes

## 🚀 Cara Penggunaan

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Basic Usage
```bash
# Validate
simpleneural validate examples/minimal.sndsl

# Compile
simpleneural compile examples/housing_regression.sndsl -o model.py

# Run
simpleneural run examples/iris_classification.sndsl

# Debug
simpleneural tokenize examples/minimal.sndsl
simpleneural ast examples/minimal.sndsl
```

### Python API
```python
from simpleneural import Compiler

# Compile from file
compiler = Compiler(verbose=True)
compiler.compile_file('model.sndsl', 'output.py')

# Compile from string
result = compiler.compile_string(dsl_code, 'source.sndsl')
if result['success']:
    print(result['code'])
```

## 📊 Statistics

### Code Metrics
- **Total Lines of Code**: ~2,500+
- **Python Modules**: 7
- **DSL Examples**: 6
- **Documentation**: 6 markdown files
- **Test Coverage**: 6 comprehensive tests

### Supported Features
- **Layer Types**: 8 (Dense, Conv2D, Dropout, Flatten, LSTM, GRU, BatchNorm, MaxPool2D)
- **Activations**: 9 (relu, sigmoid, tanh, softmax, linear, selu, elu, swish, gelu)
- **Optimizers**: 6 (adam, sgd, rmsprop, adagrad, adamw, nadam)
- **Parameters**: 15+ configurable parameters

## 🧪 Testing

### Test Suite (test_compiler.py)
```bash
python test_compiler.py
```

Tests:
1. ✅ Lexer - Tokenization
2. ✅ Parser - Syntax Analysis
3. ✅ Semantic Analyzer - Validation
4. ✅ Code Generator - Python Output
5. ✅ Full Compiler - End-to-End
6. ✅ Error Detection - Error Handling

### Example Validation
```bash
# All examples
for file in examples/*.sndsl; do
    simpleneural validate "$file"
done
```

## 🎓 Konsep Automata

### Implementasi Teori
| Komponen | Konsep | Implementasi |
|----------|--------|--------------|
| **Lexer** | DFA/NFA | Token pattern matching |
| **Lexer** | Regular Expression | Token definition |
| **Parser** | CFG (Context-Free Grammar) | Grammar rules |
| **Parser** | Recursive Descent | LL(1) parsing |
| **Semantic** | Symbol Table | Scope tracking |
| **Semantic** | Type System | Type checking |
| **CodeGen** | Template-based | AST transformation |

### Grammar (Simplified BNF)
```bnf
Program    ::= Dataset? Model+
Dataset    ::= DATASET load STRING TARGET STRING
Model      ::= MODEL STRING { Layer+ Optimizer? TrainConfig? }
Layer      ::= LAYER LayerType Parameters*
Optimizer  ::= OPTIMIZER STRING Parameters+
TrainConfig ::= TRAIN Parameters+
Parameters ::= IDENTIFIER : Value
Value      ::= STRING | INTEGER | FLOAT | BOOLEAN | TUPLE
```

## 📝 DSL Syntax

### Minimal Example
```plaintext
DATASET load "data.csv" TARGET "y"

MODEL "SimpleModel" {
    LAYER DENSE units: 32 activation: "relu"
    LAYER DENSE units: 1 activation: "linear"
    
    OPTIMIZER "adam" lr: 0.01
    TRAIN epochs: 50 batch_size: 32 validation_split: 0.2
}
```

### Advanced Example
```plaintext
DATASET load "complex_data.csv" TARGET "outcome"

MODEL "DeepNetwork" {
    LAYER DENSE units: 256 activation: "relu"
    LAYER BATCHNORM
    LAYER DROPOUT rate: 0.4
    
    LAYER LSTM units: 128 return_sequences: true
    LAYER DROPOUT rate: 0.3
    
    LAYER DENSE units: 64 activation: "relu"
    LAYER DENSE units: 1 activation: "sigmoid"
    
    OPTIMIZER "adam" lr: 0.0005
    TRAIN epochs: 150 batch_size: 64 validation_split: 0.25
}
```

## 🔧 Development

### Adding New Features

#### 1. New Layer Type
1. Add token in `lexer.py`: `KEYWORD_NEWLAYER`
2. Add pattern in `TOKEN_PATTERNS`
3. Update parser in `parser.py`: layer type checking
4. Add validation in `semantic.py`: `LAYER_PARAMS`
5. Add code generation in `codegen.py`: `generate_layer()`

#### 2. New Optimizer
1. Add to `VALID_OPTIMIZERS` in `semantic.py`
2. Add case in `generate_optimizer()` in `codegen.py`

#### 3. New Activation
1. Add to `VALID_ACTIVATIONS` in `semantic.py`

## 📚 Documentation

### Main Documentation
- [README.md](README.md) - Main documentation
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide

### Detailed Documentation (docs/)
1. **Pendahuluan**: Background, objectives, requirements
2. **Use Case**: Diagrams, specifications, user stories
3. **Arsitektur**: ERD, system architecture, class diagrams
4. **Grammar**: Token specification, CFG, semantic rules
5. **Implementasi**: Code generation, pseudocode, CLI
6. **Testing**: Test plan, examples, deployment

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Update documentation
5. Submit pull request

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 👥 Authors

SimpleNeural Team - Tugas Teori Automata & Bahasa Formal

## 🙏 Acknowledgments

- TensorFlow/Keras team
- Python community
- Automata theory course materials

---

**SimpleNeural-DSL** - Making Machine Learning Configuration Simple! 🚀
