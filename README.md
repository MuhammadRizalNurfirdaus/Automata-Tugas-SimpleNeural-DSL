# SimpleNeural-DSL

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Domain Specific Language untuk konfigurasi Machine Learning dengan sintaks sederhana dan deklaratif. Mengimplementasikan konsep teori automata dan kompilasi untuk menghasilkan kode Python yang production-ready.

## 🎯 Fitur

- **Sintaks Sederhana** - Definisi model dengan bahasa yang mudah dipahami
- **Type-Safe** - Validasi sintaks dan semantik pada compile-time  
- **Production-Ready** - Menghasilkan kode Python yang clean dan teroptimasi
- **Framework Support** - Support untuk TensorFlow/Keras
- **CLI & UI** - Command-line dan menu-driven interface
- **Kaggle Integration** - Download dan auto-generate DSL dari dataset
- **Error Messages** - Pesan error yang informatif dan actionable

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.13+
- pandas, numpy, scikit-learn

## 🚀 Installation

### Setup Virtual Environment (Recommended)

```bash
# Clone repository
git clone https://github.com/MythEclipse/Automata-Tugas-SimpleNeural-DSL.git
cd Automata-Tugas-SimpleNeural-DSL

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

# Verify installation
simpleneural --version
```

### Quick Setup Script

```bash
# Run automated setup script
bash setup_venv.sh

# Activate the environment
source venv/bin/activate
```

## 💡 Quick Start

### Interactive UI (Recommended)

```bash
python ui.py
```

**Features:**
- Load DSL file or write code directly (Option 1-2)
- 🤖 Auto-generate DSL from Kaggle dataset (Option 3)
- 📥 Download dataset from Kaggle (Option 4)
- View, tokenize, parse, validate, compile (Option 5-9)
- Compile and run directly (Option A)

### Using DSL File

**1. Create DSL File** (`model.sndsl`):

```plaintext
# SimpleNeural DSL - House Price Prediction
DATASET load "housing_data.csv" TARGET "price"

MODEL "HousePricePredictor" {
    LAYER DENSE units: 128 activation: "relu"
    LAYER DROPOUT rate: 0.3
    LAYER DENSE units: 64 activation: "relu"
    LAYER DENSE units: 1 activation: "linear"
    
    OPTIMIZER "adam" lr: 0.001
    TRAIN epochs: 100 batch_size: 32 validation_split: 0.2
}
```

**2. Compile & Run:**

```bash
# Compile to Python
simpleneural compile model.sndsl -o output.py

# Or run directly
simpleneural run model.sndsl
```

## 📖 Sintaks DSL

### Dataset Configuration

```plaintext
DATASET load "<file.csv>" TARGET "<target_column>"
```

### Model Definition

```plaintext
MODEL "<model_name>" {
    LAYER <type> <parameters...>
    ...
    OPTIMIZER "<optimizer>" <parameters...>
    TRAIN <parameters...>
}
```

### Layer Types

| Layer Type | Parameters | Example |
|------------|------------|---------|
| `DENSE` | `units`, `activation` | `LAYER DENSE units: 64 activation: "relu"` |
| `DROPOUT` | `rate` | `LAYER DROPOUT rate: 0.5` |
| `CONV2D` | `filters`, `kernel_size`, `activation` | `LAYER CONV2D filters: 32 kernel_size: (3,3)` |
| `FLATTEN` | - | `LAYER FLATTEN` |
| `LSTM` | `units`, `return_sequences` | `LAYER LSTM units: 128 return_sequences: true` |
| `GRU` | `units` | `LAYER GRU units: 64` |
| `BATCHNORM` | - | `LAYER BATCHNORM` |
| `MAXPOOL2D` | `pool_size` | `LAYER MAXPOOL2D pool_size: (2,2)` |

### Activation Functions

```plaintext
relu, sigmoid, tanh, softmax, linear, selu, elu, swish, gelu
```

### Optimizers

```plaintext
adam, sgd, rmsprop, adagrad, adamw, nadam
```

## 🌐 Kaggle Integration

Download dan auto-generate DSL dari Kaggle datasets.

### Quick Start

```bash
python ui.py
# Option 3: Auto-generate DSL from Kaggle
# Option 4: Download dataset only
```

### Setup

1. Install Kaggle CLI: `pip install kaggle`
2. Get API credentials from [kaggle.com/account](https://www.kaggle.com/account)
3. UI will guide setup automatically

See [KAGGLE_QUICKSTART.md](KAGGLE_QUICKSTART.md) for detailed guide.

## 🔧 CLI Commands

```bash
# Compile DSL to Python
simpleneural compile <input.sndsl> [-o output.py] [-v]

# Validate DSL syntax
simpleneural validate <input.sndsl>

# Compile and run
simpleneural run <input.sndsl> [-o output.py]

# Debug tools
simpleneural tokenize <input.sndsl>  # View tokens
simpleneural ast <input.sndsl>       # View AST
```

## 📁 Struktur Project

```
automata/
├── simpleneural/           # Package utama
│   ├── __init__.py
│   ├── __main__.py        # Entry point
│   ├── lexer.py           # Lexical analyzer
│   ├── parser.py          # Syntax parser
│   ├── semantic.py        # Semantic analyzer
│   ├── codegen.py         # Code generator
│   ├── compiler.py        # Main compiler
│   └── cli.py             # CLI interface
│
├── examples/              # Contoh file DSL
│   ├── minimal.sndsl
│   ├── housing_regression.sndsl
│   ├── iris_classification.sndsl
│   ├── deep_network.sndsl
│   └── lstm_timeseries.sndsl
│
├── docs/                  # Dokumentasi lengkap
│   ├── 01-pendahuluan.md
│   ├── 02-use-case.md
│   ├── 03-arsitektur.md
│   ├── 04-grammar-token.md
│   ├── 05-implementasi.md
│   └── 06-testing-examples.md
│
├── setup.py
├── requirements.txt
└── README.md
```

## 🎓 Konsep Automata

| Komponen | Teknik | Penerapan |
|----------|--------|-----------|
| **Lexer** | Finite Automata (DFA/NFA) | Pattern matching untuk token |
| **Lexer** | Regular Expression | Definisi pattern token |
| **Parser** | Context-Free Grammar | Aturan sintaks bahasa |
| **Parser** | Recursive Descent (LL1) | Algoritma parsing |
| **Semantic** | Symbol Table | Tracking definisi dan scope |
| **Semantic** | Type System | Validasi tipe data |
| **CodeGen** | Template-Based | Transformasi AST ke Python |

## 📚 Dokumentasi

Dokumentasi lengkap di folder [`docs/`](docs/):

1. [Pendahuluan](docs/01-pendahuluan.md) - Latar belakang dan tujuan
2. [Use Case](docs/02-use-case.md) - Diagram dan spesifikasi
3. [Arsitektur](docs/03-arsitektur.md) - ERD dan class diagrams
4. [Grammar & Token](docs/04-grammar-token.md) - Spesifikasi lexer dan CFG
5. [Implementasi](docs/05-implementasi.md) - Detail implementasi
6. [Testing](docs/06-testing-examples.md) - Test dan contoh

## 🧪 Testing

```bash
# Validate examples
simpleneural validate examples/minimal.sndsl

# Compile with verbose
simpleneural compile examples/housing_regression.sndsl -v

# View tokenization
simpleneural tokenize examples/iris_classification.sndsl

# View AST
simpleneural ast examples/deep_network.sndsl
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👥 Authors

SimpleNeural Team - Tugas Teori Automata & Bahasa Formal

---

**SimpleNeural-DSL** - Making Machine Learning Configuration Simple and Type-Safe! 🚀
