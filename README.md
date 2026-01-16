# SimpleNeural-DSL: Domain Specific Language untuk Konfigurasi Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SimpleNeural-DSL adalah Domain Specific Language (DSL) yang memungkinkan Anda mendefinisikan dan mengkonfigurasi model Machine Learning dengan sintaks sederhana dan deklaratif. Proyek ini mengimplementasikan konsep-konsep teori automata dan kompilasi untuk menghasilkan kode Python yang production-ready.

## 🎯 Fitur Utama

- **Sintaks Sederhana**: Definisi model dengan bahasa yang mudah dipahami
- **Type-Safe**: Validasi sintaks dan semantik pada compile-time
- **Production-Ready**: Menghasilkan kode Python yang clean dan teroptimasi
- **Framework Support**: Support untuk TensorFlow/Keras
- **CLI Tools**: Command-line interface yang lengkap
- **Error Messages**: Pesan error yang informatif dan actionable

## 📋 Requirements

- Python 3.8 atau lebih tinggi
- TensorFlow 2.13+
- pandas, numpy, scikit-learn

## 🚀 Installation

### Dari Source

```bash
# Clone repository
git clone https://github.com/MythEclipse/Automata-Tugas-SimpleNeural-DSL.git
cd Automata-Tugas-SimpleNeural-DSL

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Cek Instalasi

```bash
simpleneural --version
```

## 💡 Quick Start

### 1. Buat File DSL

Buat file `model.sndsl`:

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

### 2. Compile ke Python

```bash
simpleneural compile model.sndsl -o output.py
```

### 3. Jalankan Model

```bash
python output.py
```

Atau compile dan run sekaligus:

```bash
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

## 🔧 CLI Commands

### Compile

Compile DSL file ke Python:

```bash
simpleneural compile <input.sndsl> [-o output.py] [-v]
```

### Validate

Validasi DSL tanpa generate code:

```bash
simpleneural validate <input.sndsl>
```

### Run

Compile dan jalankan:

```bash
simpleneural run <input.sndsl> [-o output.py]
```

### Debug Tools

Lihat tokens (untuk debugging):

```bash
simpleneural tokenize <input.sndsl>
```

Lihat AST (untuk debugging):

```bash
simpleneural ast <input.sndsl>
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

## 🎓 Konsep Automata yang Digunakan

Proyek ini mengimplementasikan berbagai konsep teori automata dan kompilasi:

| Komponen | Teknik Automata | Penerapan |
|----------|----------------|-----------|
| **Lexer** | Finite Automata (DFA/NFA) | Pattern matching untuk token recognition |
| **Lexer** | Regular Expression | Definisi pattern token |
| **Parser** | Context-Free Grammar | Aturan sintaks bahasa |
| **Parser** | Recursive Descent | Algoritma parsing LL(1) |
| **Semantic** | Symbol Table | Tracking definisi dan scope |
| **Semantic** | Type System | Validasi tipe data |
| **CodeGen** | Template-Based | Transformasi AST ke Python |

## 📚 Dokumentasi Lengkap

Dokumentasi lengkap tersedia di folder `docs/`:

1. [Pendahuluan](docs/01-pendahuluan.md) - Latar belakang dan tujuan
2. [Use Case Analysis](docs/02-use-case.md) - Diagram dan spesifikasi
3. [Arsitektur Sistem](docs/03-arsitektur.md) - ERD dan class diagrams
4. [Grammar & Token](docs/04-grammar-token.md) - Spesifikasi lexer dan CFG
5. [Implementasi](docs/05-implementasi.md) - Detail implementasi
6. [Testing & Examples](docs/06-testing-examples.md) - Test dan contoh

## 🧪 Testing

Jalankan test pada contoh file:

```bash
# Test minimal example
simpleneural validate examples/minimal.sndsl

# Test dengan verbose output
simpleneural compile examples/housing_regression.sndsl -v

# Test tokenization
simpleneural tokenize examples/iris_classification.sndsl

# Test AST generation
simpleneural ast examples/deep_network.sndsl
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- SimpleNeural Team
- Tugas Teori Automata & Bahasa Formal

## 🙏 Acknowledgments

- TensorFlow/Keras team
- Teori Automata & Kompilasi course materials
- Open source Python community

## 📞 Contact

For questions and support, please open an issue on GitHub.

---

**SimpleNeural-DSL** - Making Machine Learning Configuration Simple and Type-Safe! 🚀
