# SimpleNeural-DSL: Quick Start Guide

## Instalasi

```bash
# Clone repository
git clone https://github.com/MythEclipse/Automata-Tugas-SimpleNeural-DSL.git
cd Automata-Tugas-SimpleNeural-DSL

# Install dependencies
pip install -r requirements.txt

# Install package (development mode)
pip install -e .
```

## Penggunaan Dasar

### 1. Membuat File DSL

Buat file `mymodel.sndsl`:

```plaintext
DATASET load "data.csv" TARGET "target_column"

MODEL "MyFirstModel" {
    LAYER DENSE units: 64 activation: "relu"
    LAYER DENSE units: 32 activation: "relu"
    LAYER DENSE units: 1 activation: "linear"
    
    OPTIMIZER "adam" lr: 0.01
    TRAIN epochs: 100 batch_size: 32 validation_split: 0.2
}
```

### 2. Validasi File

```bash
simpleneural validate mymodel.sndsl
```

Output:
```
✓ Lexical analysis passed (XX tokens)
✓ Syntax analysis passed
✓ Semantic analysis passed
✅ File is valid!
```

### 3. Compile ke Python

```bash
simpleneural compile mymodel.sndsl -o model.py
```

Output:
```
✅ Compilation successful!
📝 Output written to: model.py
```

### 4. Jalankan Model

```bash
python model.py
```

Atau compile dan run sekaligus:

```bash
simpleneural run mymodel.sndsl
```

## Command Reference

### compile
Compile DSL file ke Python code:
```bash
simpleneural compile <input.sndsl> [-o output.py] [-v]
```

Options:
- `-o, --output`: Output file name (default: same as input with .py extension)
- `-v, --verbose`: Show detailed compilation stages

### validate
Validate DSL file tanpa generate code:
```bash
simpleneural validate <input.sndsl> [-v]
```

### run
Compile dan jalankan generated Python code:
```bash
simpleneural run <input.sndsl> [-o output.py] [-v]
```

### tokenize (Debug)
Show token list untuk debugging:
```bash
simpleneural tokenize <input.sndsl>
```

### ast (Debug)
Show Abstract Syntax Tree untuk debugging:
```bash
simpleneural ast <input.sndsl>
```

## Contoh Lengkap

### Example 1: Regression

```plaintext
# File: house_price.sndsl
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
```

Compile:
```bash
simpleneural compile house_price.sndsl
```

### Example 2: Classification

```plaintext
# File: iris.sndsl
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
```

### Example 3: Time Series dengan LSTM

```plaintext
# File: timeseries.sndsl
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
```

## Sintaks DSL

### Dataset Declaration
```plaintext
DATASET load "<filename.csv>" TARGET "<column_name>"
```

### Model Declaration
```plaintext
MODEL "<model_name>" {
    LAYER <type> <params...>
    ...
    OPTIMIZER "<type>" <params...>
    TRAIN <params...>
}
```

### Layer Types

#### DENSE Layer
```plaintext
LAYER DENSE units: <int> activation: "<function>"
```
Parameters:
- `units`: Jumlah neuron (required, positive integer)
- `activation`: Fungsi aktivasi (required)

#### DROPOUT Layer
```plaintext
LAYER DROPOUT rate: <float>
```
Parameters:
- `rate`: Dropout rate 0.0-1.0 (required)

#### LSTM Layer
```plaintext
LAYER LSTM units: <int> [return_sequences: <bool>]
```
Parameters:
- `units`: Jumlah units (required)
- `return_sequences`: Return sequences (optional, default: false)

#### GRU Layer
```plaintext
LAYER GRU units: <int> [return_sequences: <bool>]
```

#### BATCHNORM Layer
```plaintext
LAYER BATCHNORM
```

#### FLATTEN Layer
```plaintext
LAYER FLATTEN
```

#### CONV2D Layer
```plaintext
LAYER CONV2D filters: <int> kernel_size: (<int>,<int>) [activation: "<function>"]
```

#### MAXPOOL2D Layer
```plaintext
LAYER MAXPOOL2D pool_size: (<int>,<int>)
```

### Activation Functions
- `relu` - Rectified Linear Unit
- `sigmoid` - Sigmoid function
- `tanh` - Hyperbolic tangent
- `softmax` - Softmax (untuk klasifikasi multi-class)
- `linear` - Linear activation
- `selu` - Scaled Exponential Linear Unit
- `elu` - Exponential Linear Unit
- `swish` - Swish activation
- `gelu` - Gaussian Error Linear Unit

### Optimizer Configuration
```plaintext
OPTIMIZER "<type>" lr: <float> [<other_params>]
```

Optimizer types:
- `adam` - Adam optimizer (recommended)
- `sgd` - Stochastic Gradient Descent (supports `momentum` parameter)
- `rmsprop` - RMSprop
- `adagrad` - Adagrad
- `adamw` - AdamW
- `nadam` - Nadam

Parameters:
- `lr`: Learning rate (required, 0.0001-1.0)
- `momentum`: Momentum untuk SGD (optional, 0.0-1.0)

### Training Configuration
```plaintext
TRAIN epochs: <int> batch_size: <int> [validation_split: <float>]
```

Parameters:
- `epochs`: Jumlah epochs (required, positive integer)
- `batch_size`: Batch size (required, positive integer)
- `validation_split`: Validation split ratio (optional, 0.0-1.0)

## Error Messages

### Lexical Errors
```
Lexical Error at line X, column Y: Unexpected character: 'Z'
```
- Karakter tidak dikenali
- Periksa sintaks DSL

### Syntax Errors
```
Parse Error at line X, column Y: Expected TOKEN_TYPE, got TOKEN_VALUE
```
- Kesalahan struktur sintaks
- Periksa tanda kurung, tanda titik dua, dll

### Semantic Errors
```
Semantic Error at line X, column Y: [description]
```
Contoh:
- `Invalid activation function 'xyz'`
- `Parameter 'units' must be a positive integer`
- `Dropout rate must be between 0.0 and 1.0`
- `Model must have at least one layer`

## Tips & Best Practices

1. **Gunakan validation_split**: Selalu gunakan validation split untuk monitoring training
2. **Dropout untuk Regularization**: Tambahkan dropout layer untuk mengurangi overfitting
3. **Learning Rate**: Mulai dengan 0.001-0.01 untuk kebanyakan kasus
4. **Batch Size**: 32 atau 64 adalah pilihan yang baik untuk mulai
5. **Activation Functions**:
   - Hidden layers: `relu` atau `selu`
   - Output regression: `linear`
   - Output binary classification: `sigmoid`
   - Output multi-class: `softmax`

## Troubleshooting

### File not found error
```
❌ Error: File 'mymodel.sndsl' not found
```
Solution: Pastikan file ada di direktori yang benar

### Dataset not found saat run
```
FileNotFoundError: Dataset 'data.csv' not found
```
Solution: Pastikan file CSV ada di direktori tempat menjalankan script

### Invalid token
```
Lexical Error: Unexpected character
```
Solution: Periksa sintaks, pastikan tidak ada typo

### Semantic validation error
Solution: Ikuti saran dari error message untuk memperbaiki konfigurasi

## Contoh Lengkap dengan Data

Lihat folder `examples/` untuk contoh lengkap:
- `minimal.sndsl` - Contoh paling sederhana
- `housing_regression.sndsl` - Prediksi harga rumah
- `iris_classification.sndsl` - Klasifikasi Iris
- `deep_network.sndsl` - Deep neural network
- `lstm_timeseries.sndsl` - Time series dengan LSTM

## Dokumentasi Lengkap

Untuk dokumentasi lengkap, lihat folder `docs/`:
1. [Pendahuluan](docs/01-pendahuluan.md)
2. [Use Case Analysis](docs/02-use-case.md)
3. [Arsitektur Sistem](docs/03-arsitektur.md)
4. [Grammar & Token](docs/04-grammar-token.md)
5. [Implementasi](docs/05-implementasi.md)
6. [Testing & Examples](docs/06-testing-examples.md)
