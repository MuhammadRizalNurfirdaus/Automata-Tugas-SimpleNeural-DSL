#!/usr/bin/env python3
"""
SimpleNeural-DSL Interactive UI
Simple and user-friendly interface for DSL compilation
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from simpleneural.compiler import Compiler
from simpleneural.lexer import Lexer
from simpleneural.parser import Parser


class SimpleNeuralUI:
    """Simple interactive UI for SimpleNeural-DSL"""
    
    def __init__(self):
        self.compiler = Compiler()
        self.current_file = None
        self.current_code = None  # For direct DSL input
        self.input_mode = None  # 'file' or 'direct'
        self.kaggle_available = self._check_kaggle_available()
        self.kaggle_config = self._load_kaggle_config() if self.kaggle_available else None
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name != 'nt' else 'cls')
        
    def print_header(self):
        """Print application header"""
        print("=" * 70)
        print("  🧠 SimpleNeural-DSL - Machine Learning Model Compiler")
        print("=" * 70)
        print()
        
    def print_menu(self):
        """Print main menu"""
        print("📋 MENU UTAMA:")
        print("-" * 70)
        print("  1. 📂 Load DSL File")
        print("  2. ✍️  Write DSL Code (Direct Input)")
        kaggle_status = "✅" if self.kaggle_available else "⚠️ "
        print(f"  3. 🤖 Auto-Generate DSL from Kaggle {kaggle_status}")
        print(f"  4. 📥 Download Dataset from Kaggle {kaggle_status}")
        print("  5. 🔍 View File/Code Content")
        print("  6. 🔤 Show Tokens (Lexical Analysis)")
        print("  7. 🌳 Show AST (Syntax Analysis)")
        print("  8. ✅ Validate (Semantic Analysis)")
        print("  9. ⚙️  Compile to Python")
        print("  A. 🚀 Compile & Run")
        print("  B. 📚 Show Examples")
        print("  C. ❓ Help")
        print("  0. 🚪 Exit")
        print("-" * 70)
        print()
        
    def load_file(self):
        """Load DSL file"""
        print("\n📂 LOAD DSL FILE")
        print("-" * 70)
        
        # Show available examples
        examples_dir = Path(__file__).parent / "examples"
        if examples_dir.exists():
            print("\n📚 Available examples:")
            examples = sorted(examples_dir.glob("*.sndsl"))
            for i, ex in enumerate(examples, 1):
                print(f"  {i}. {ex.name}")
            print()
        
        file_path = input("Enter file path (or number for example): ").strip()
        
        # Handle numeric input for examples
        if file_path.isdigit() and examples_dir.exists():
            idx = int(file_path) - 1
            if 0 <= idx < len(examples):
                file_path = str(examples[idx])
        
        if not os.path.exists(file_path):
            print(f"❌ Error: File '{file_path}' not found!")
            return
            
        self.current_file = file_path
        self.current_code = None
        self.input_mode = 'file'
        print(f"✅ Loaded: {file_path}")
        
    def write_dsl_code(self):
        """Write DSL code directly"""
        print("\n✍️  WRITE DSL CODE (DIRECT INPUT)")
        print("-" * 70)
        print("\n💡 Tips:")
        print("  • Write your DSL code line by line")
        print("  • Type 'END' on a new line when finished")
        print("  • Type 'CANCEL' to cancel input")
        print("  • Type 'TEMPLATE' to load a template")
        print()
        
        choice = input("Load template? (y/n): ").strip().lower()
        
        if choice == 'y':
            template = '''# SimpleNeural DSL - Quick Template
DATASET load "your_data.csv" TARGET "target_column"

MODEL "YourModel" {
    LAYER DENSE units: 64 activation: "relu"
    LAYER DROPOUT rate: 0.2
    LAYER DENSE units: 1 activation: "linear"
    
    OPTIMIZER "adam" lr: 0.001
    TRAIN epochs: 50 batch_size: 32
}'''
            print("\n📋 Template loaded. Edit as needed:")
            print("-" * 70)
            print(template)
            print("-" * 70)
            print()
            
            if input("Use this template? (y/n): ").strip().lower() == 'y':
                self.current_code = template
                self.current_file = None
                self.input_mode = 'direct'
                print("✅ Template loaded! You can now validate or compile.")
                return
        
        print("\n📝 Enter your DSL code (type 'END' when done):")
        print("-" * 70)
        
        lines = []
        line_num = 1
        
        while True:
            try:
                line = input(f"{line_num:3d} | ")
                
                if line.strip().upper() == 'END':
                    break
                elif line.strip().upper() == 'CANCEL':
                    print("❌ Input cancelled.")
                    return
                    
                lines.append(line)
                line_num += 1
                
            except EOFError:
                break
        
        if not lines:
            print("❌ No code entered!")
            return
            
        self.current_code = '\n'.join(lines)
        self.current_file = None
        self.input_mode = 'direct'
        
        print()
        print("✅ DSL code saved!")
        print(f"📊 Total lines: {len(lines)}")
        print()
        print("💡 You can now:")
        print("  • View the code (option 3)")
        print("  • Show tokens (option 4)")
        print("  • Validate (option 6)")
        print("  • Compile (option 7)")
    
    def _check_kaggle_available(self):
        """Check if Kaggle CLI is available"""
        try:
            result = subprocess.run(['kaggle', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _load_kaggle_config(self):
        """Load Kaggle API configuration"""
        config_file = Path.home() / ".kaggle" / "kaggle.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return None
    
    def _save_kaggle_config(self, username, key):
        """Save Kaggle API configuration"""
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        
        config_file = kaggle_dir / "kaggle.json"
        config_data = {"username": username, "key": key}
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Set proper permissions (600)
        os.chmod(config_file, 0o600)
        
        self.kaggle_config = config_data
        print("✅ Kaggle API credentials saved!")
    
    def download_kaggle_dataset(self):
        """Download dataset from Kaggle"""
        print("\n📥 DOWNLOAD DATASET FROM KAGGLE")
        print("-" * 70)
        
        # Check if kaggle API is available
        if not self.kaggle_available:
            print("\n❌ Kaggle CLI not found!")
            print("\n📦 INSTALLATION OPTIONS (Arch Linux):")
            print("-" * 70)
            print("\n1️⃣  Virtual Environment (RECOMMENDED):")
            print("   python -m venv ~/venv-automata")
            print("   source ~/venv-automata/bin/activate")
            print("   pip install kaggle")
            print("   python ui.py")
            print("\n2️⃣  Pipx (For CLI tools):")
            print("   sudo pacman -S python-pipx")
            print("   pipx install kaggle")
            print("\n3️⃣  System-wide (Not recommended):")
            print("   pip install kaggle --break-system-packages")
            print("-" * 70)
            return
        
        # Check/setup API credentials
        if not self.kaggle_config:
            print("\n⚠️  Kaggle API credentials not found!")
            print("\n💡 You need Kaggle API credentials to download datasets.")
            print("   Get them from: https://www.kaggle.com/account")
            print("   (Account → Create New API Token)")
            print()
            
            setup = input("Setup now? (y/n): ").strip().lower()
            if setup != 'y':
                return
            
            print("\n📝 Enter your Kaggle credentials:")
            username = input("  Username: ").strip()
            key = input("  API Key: ").strip()
            
            if username and key:
                self._save_kaggle_config(username, key)
            else:
                print("❌ Invalid credentials!")
                return
        
        print("\n📚 Popular datasets:")
        print("  1. uciml/iris - Iris Species Dataset")
        print("  2. mlg-ulb/creditcardfraud - Credit Card Fraud")
        print("  3. heptapod/titanic - Titanic Dataset")
        print("  4. uciml/pima-indians-diabetes-database - Diabetes")
        print("  5. uciml/breast-cancer-wisconsin-data - Breast Cancer")
        print("  6. Enter custom dataset path")
        print()
        
        choice = input("Choose dataset (1-6): ").strip()
        
        # Map popular datasets
        popular_datasets = {
            '1': 'uciml/iris',
            '2': 'mlg-ulb/creditcardfraud',
            '3': 'heptapod/titanic',
            '4': 'uciml/pima-indians-diabetes-database',
            '5': 'uciml/breast-cancer-wisconsin-data'
        }
        
        if choice in popular_datasets:
            dataset_path = popular_datasets[choice]
        elif choice == '6':
            dataset_path = input("Enter dataset path (owner/dataset-name): ").strip()
        else:
            print("❌ Invalid choice!")
            return
        
        if not dataset_path:
            print("❌ No dataset specified!")
            return
        
        # Create datasets directory
        datasets_dir = Path(__file__).parent / "datasets"
        datasets_dir.mkdir(exist_ok=True)
        
        print(f"\n⏳ Downloading {dataset_path}...")
        print("-" * 70)
        
        try:
            # Download dataset
            result = subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', dataset_path, 
                 '-p', str(datasets_dir), '--unzip'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                print("✅ Dataset downloaded successfully!")
                print(f"\n📁 Location: {datasets_dir}")
                
                # List downloaded files
                csv_files = list(datasets_dir.glob("*.csv"))
                if csv_files:
                    print("\n📄 CSV files found:")
                    for csv_file in csv_files:
                        size = csv_file.stat().st_size / 1024  # KB
                        print(f"  • {csv_file.name} ({size:.1f} KB)")
                else:
                    print("\n⚠️  No CSV files found in dataset")
                
            else:
                print(f"❌ Download failed!")
                if result.stderr:
                    print(f"\nError: {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            print("❌ Download timeout (>5 minutes)")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def auto_generate_from_kaggle(self):
        """Auto-generate DSL from Kaggle dataset"""
        print("\n🤖 AUTO-GENERATE DSL FROM KAGGLE")
        print("=" * 70)
        print("\n💡 This feature will:")
        print("   1. Download dataset from Kaggle")
        print("   2. Analyze the dataset automatically")
        print("   3. Generate DSL file based on analysis")
        print("   4. Ready to compile and run!")
        print("-" * 70)
        
        # Check kaggle CLI
        if not self.kaggle_available:
            print("\n❌ Kaggle CLI not found!")
            print("\n📦 INSTALLATION OPTIONS (Arch Linux):")
            print("-" * 70)
            print("\n1️⃣  Virtual Environment (RECOMMENDED):")
            print("   python -m venv ~/venv-automata")
            print("   source ~/venv-automata/bin/activate")
            print("   pip install kaggle")
            print("   python ui.py")
            print("\n2️⃣  Pipx (For CLI tools):")
            print("   sudo pacman -S python-pipx")
            print("   pipx install kaggle")
            print("\n3️⃣  System-wide (Not recommended):")
            print("   pip install kaggle --break-system-packages")
            print("-" * 70)
            print("\n💡 After installation, restart this program.")
            return
        
        # Check/setup API credentials
        if not self.kaggle_config:
            print("\n⚠️  Kaggle API credentials not found!")
            print("\n💡 You need Kaggle API credentials.")
            print("   Get them from: https://www.kaggle.com/account")
            print("   (Account → Create New API Token)")
            print()
            
            setup = input("Setup now? (y/n): ").strip().lower()
            if setup != 'y':
                return
            
            print("\n📝 Enter your Kaggle credentials:")
            username = input("  Username: ").strip()
            key = input("  API Key: ").strip()
            
            if username and key:
                self._save_kaggle_config(username, key)
            else:
                print("❌ Invalid credentials!")
                return
        
        # Select dataset
        print("\n📚 Popular ML datasets:")
        print("  1. uciml/iris - Iris Species (Classification)")
        print("  2. heptapod/titanic - Titanic Survival (Classification)")
        print("  3. uciml/pima-indians-diabetes-database - Diabetes (Classification)")
        print("  4. uciml/breast-cancer-wisconsin-data - Breast Cancer (Classification)")
        print("  5. mlg-ulb/creditcardfraud - Credit Card Fraud (Classification)")
        print("  6. Enter custom dataset path")
        print()
        
        choice = input("Choose dataset (1-6): ").strip()
        
        popular_datasets = {
            '1': ('uciml/iris', 'Iris', 'Species'),
            '2': ('heptapod/titanic', 'Titanic', 'Survived'),
            '3': ('uciml/pima-indians-diabetes-database', 'Diabetes', 'Outcome'),
            '4': ('uciml/breast-cancer-wisconsin-data', 'BreastCancer', 'diagnosis'),
            '5': ('mlg-ulb/creditcardfraud', 'CreditFraud', 'Class')
        }
        
        if choice in popular_datasets:
            dataset_path, model_name, target_col = popular_datasets[choice]
        elif choice == '6':
            dataset_path = input("Enter dataset path (owner/dataset-name): ").strip()
            model_name = input("Model name: ").strip() or "AutoModel"
            target_col = input("Target column name: ").strip()
        else:
            print("❌ Invalid choice!")
            return
        
        if not dataset_path:
            print("❌ No dataset specified!")
            return
        
        # Create datasets directory
        datasets_dir = Path(__file__).parent / "datasets"
        datasets_dir.mkdir(exist_ok=True)
        
        print(f"\n⏳ Step 1/4: Downloading {dataset_path}...")
        print("-" * 70)
        
        try:
            # Download dataset
            result = subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', dataset_path, 
                 '-p', str(datasets_dir), '--unzip'],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                print(f"❌ Download failed!")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                return
            
            print("✅ Download complete!")
            
            # Find CSV files
            csv_files = list(datasets_dir.glob("*.csv"))
            if not csv_files:
                print("❌ No CSV files found!")
                return
            
            # Use first CSV or let user choose
            if len(csv_files) == 1:
                csv_file = csv_files[0]
            else:
                print(f"\n📄 Found {len(csv_files)} CSV files:")
                for i, cf in enumerate(csv_files, 1):
                    print(f"  {i}. {cf.name}")
                idx = input("\nChoose file (1-{}): ".format(len(csv_files))).strip()
                if idx.isdigit() and 1 <= int(idx) <= len(csv_files):
                    csv_file = csv_files[int(idx) - 1]
                else:
                    csv_file = csv_files[0]
            
            print(f"\n⏳ Step 2/4: Analyzing {csv_file.name}...")
            print("-" * 70)
            
            # Analyze dataset
            analysis = self._analyze_dataset(csv_file, target_col)
            
            if not analysis:
                print("❌ Analysis failed!")
                return
            
            print("✅ Analysis complete!")
            print(f"   • Rows: {analysis['rows']}")
            print(f"   • Columns: {analysis['num_columns']}")
            print(f"   • Target: {analysis['target_column']}")
            print(f"   • Task: {analysis['task_type']}")
            print(f"   • Classes: {analysis.get('num_classes', 'N/A')}")
            
            print(f"\n⏳ Step 3/4: Generating DSL code...")
            print("-" * 70)
            
            # Generate DSL
            dsl_code = self._generate_dsl(csv_file.name, model_name, analysis)
            
            if not dsl_code:
                print("❌ DSL generation failed!")
                return
            
            print("✅ DSL generated!")
            
            # Save DSL file
            dsl_filename = f"auto_generated_{model_name.lower()}.sndsl"
            dsl_path = Path(__file__).parent / "examples" / dsl_filename
            
            with open(dsl_path, 'w') as f:
                f.write(dsl_code)
            
            print(f"\n⏳ Step 4/4: Saving...")
            print("-" * 70)
            print(f"✅ DSL file saved: {dsl_filename}")
            
            # Load the generated file
            self.current_file = str(dsl_path)
            self.current_code = None
            self.input_mode = 'file'
            
            print("\n" + "=" * 70)
            print("🎉 AUTO-GENERATION COMPLETE!")
            print("=" * 70)
            print(f"\n📄 Generated DSL:")
            print("-" * 70)
            for i, line in enumerate(dsl_code.split('\n'), 1):
                print(f"{i:3d} | {line}")
            print("-" * 70)
            print(f"\n💡 Next steps:")
            print("   • Option 8: Validate the DSL")
            print("   • Option 9: Compile to Python")
            print("   • Option A: Compile & Run (train the model!)")
            
        except subprocess.TimeoutExpired:
            print("❌ Download timeout (>5 minutes)")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_dataset(self, csv_path, target_col=None):
        """Analyze dataset to determine structure and task type"""
        try:
            import pandas as pd
            import numpy as np
            
            # Load dataset
            df = pd.read_csv(csv_path)
            
            # Basic info
            rows, cols = df.shape
            column_names = df.columns.tolist()
            
            # Find target column (case-insensitive)
            actual_target = None
            if target_col:
                for col in column_names:
                    if col.lower() == target_col.lower():
                        actual_target = col
                        break
            
            # If target not found, guess it
            if not actual_target:
                # Common target names
                common_targets = ['target', 'label', 'class', 'y', 'output', 
                                'price', 'species', 'survived', 'outcome', 'diagnosis']
                for col in column_names:
                    if col.lower() in common_targets:
                        actual_target = col
                        break
                
                # If still not found, use last column
                if not actual_target:
                    actual_target = column_names[-1]
            
            # Determine task type
            target_data = df[actual_target]
            task_type = 'classification'
            num_classes = 0
            
            if pd.api.types.is_numeric_dtype(target_data):
                unique_values = target_data.nunique()
                if unique_values <= 20:  # Classification
                    task_type = 'classification'
                    num_classes = unique_values
                else:  # Regression
                    task_type = 'regression'
            else:  # Categorical = Classification
                task_type = 'classification'
                num_classes = target_data.nunique()
            
            # Get feature columns (exclude target and ID-like columns)
            feature_cols = []
            for col in column_names:
                if col == actual_target:
                    continue
                # Skip ID columns
                if col.lower() in ['id', 'index', 'unnamed: 0']:
                    continue
                feature_cols.append(col)
            
            return {
                'rows': rows,
                'num_columns': cols,
                'columns': column_names,
                'target_column': actual_target,
                'feature_columns': feature_cols,
                'task_type': task_type,
                'num_classes': num_classes,
                'has_missing': df.isnull().sum().sum() > 0
            }
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return None
    
    def _generate_dsl(self, csv_filename, model_name, analysis):
        """Generate DSL code based on dataset analysis"""
        try:
            target = analysis['target_column']
            task_type = analysis['task_type']
            num_classes = analysis.get('num_classes', 2)
            num_features = len(analysis['feature_columns'])
            
            # Determine architecture with more layers for better accuracy
            if task_type == 'classification':
                if num_classes == 2:
                    # Binary classification
                    output_units = 1
                    output_activation = 'sigmoid'
                else:
                    # Multi-class classification
                    output_units = num_classes
                    output_activation = 'softmax'
                
                # Dynamic hidden layers for high accuracy (99% target)
                if num_features < 10:
                    hidden_units = [64, 32, 16]  # Deeper network
                elif num_features < 30:
                    hidden_units = [128, 64, 32]
                elif num_features < 50:
                    hidden_units = [256, 128, 64, 32]
                else:
                    hidden_units = [512, 256, 128, 64, 32]  # Very deep for complex data
            
            else:  # Regression
                output_units = 1
                output_activation = 'linear'
                
                # Dynamic architecture for regression
                if num_features < 10:
                    hidden_units = [128, 64, 32]
                elif num_features < 30:
                    hidden_units = [256, 128, 64]
                elif num_features < 50:
                    hidden_units = [512, 256, 128, 64]
                else:
                    hidden_units = [512, 256, 128, 64, 32]  # Deep network for complex patterns
            
            # Generate DSL code with dynamic architecture
            dsl_lines = []
            dsl_lines.append(f"# Auto-generated DSL for {model_name}")
            dsl_lines.append(f"# Dataset: {csv_filename}")
            dsl_lines.append(f"# Task: {task_type.capitalize()}")
            if task_type == 'classification':
                dsl_lines.append(f"# Classes: {num_classes}")
            dsl_lines.append(f"# Features: {num_features}")
            dsl_lines.append(f"# Target Accuracy: 99%")
            dsl_lines.append(f"# Early Stopping: Enabled")
            dsl_lines.append("")
            # Use datasets/ path for the DATASET load command
            dataset_path = f"datasets/{csv_filename}"
            dsl_lines.append(f'DATASET load "{dataset_path}" TARGET "{target}"')
            dsl_lines.append("")
            dsl_lines.append(f'MODEL "{model_name}" {{')
            
            # Add hidden layers with dropout and batch normalization for better performance
            for i, units in enumerate(hidden_units):
                dsl_lines.append(f'    LAYER DENSE units: {units} activation: "relu"')
                if i < len(hidden_units) - 1:  # Add dropout except for last hidden layer
                    dropout_rate = 0.3 if task_type == 'classification' else 0.2
                    dsl_lines.append(f'    LAYER DROPOUT rate: {dropout_rate}')
            
            # Output layer
            dsl_lines.append(f'    LAYER DENSE units: {output_units} activation: "{output_activation}"')
            dsl_lines.append("")
            
            # Optimizer with adjusted learning rate
            lr = 0.001 if task_type == 'classification' else 0.01
            dsl_lines.append(f'    OPTIMIZER "adam" lr: {lr}')
            
            # Training params with more epochs and validation split for early stopping
            # Set high epochs but will stop early if no improvement
            epochs = 500 if task_type == 'classification' else 300
            batch_size = 32 if analysis['rows'] > 1000 else 16
            validation_split = 0.2
            dsl_lines.append(f'    TRAIN epochs: {epochs} batch_size: {batch_size} validation_split: {validation_split}')
            dsl_lines.append('}')
            
            return '\n'.join(dsl_lines)
            
        except Exception as e:
            print(f"❌ DSL generation error: {e}")
            return None
        
    def view_file(self):
        """View current file or code content"""
        if not self.current_file and not self.current_code:
            print("❌ No file or code loaded! Please load a file or write DSL code first.")
            return
            
        if self.input_mode == 'file':
            print(f"\n📄 FILE CONTENT: {self.current_file}")
            print("-" * 70)
            
            with open(self.current_file, 'r') as f:
                content = f.read()
        else:
            print(f"\n📄 DSL CODE (Direct Input)")
            print("-" * 70)
            content = self.current_code
            
        # Show with line numbers
        for i, line in enumerate(content.split('\n'), 1):
            print(f"{i:3d} | {line}")
        print()
        
    def show_tokens(self):
        """Show lexical analysis (tokens)"""
        if not self.current_file and not self.current_code:
            print("❌ No file or code loaded! Please load a file or write DSL code first.")
            return
            
        source = self.current_file if self.input_mode == 'file' else "Direct Input"
        print(f"\n🔤 LEXICAL ANALYSIS: {source}")
        print("-" * 70)
        
        try:
            if self.input_mode == 'file':
                with open(self.current_file, 'r') as f:
                    code = f.read()
            else:
                code = self.current_code
                
            lexer = Lexer()
            tokens = lexer.tokenize(code)
            
            print(f"\n📊 Total tokens: {len(tokens)}")
            print()
            
            for i, token in enumerate(tokens, 1):
                print(f"{i:3d}. {token.type:20s} '{token.value:20s}' (line {token.line}, col {token.column})")
                
            print(f"\n✅ Lexical analysis completed successfully!")
            
        except Exception as e:
            print(f"❌ Lexical Error: {str(e)}")
            
    def show_ast(self):
        """Show syntax analysis (AST)"""
        if not self.current_file and not self.current_code:
            print("❌ No file or code loaded! Please load a file or write DSL code first.")
            return
            
        source = self.current_file if self.input_mode == 'file' else "Direct Input"
        print(f"\n🌳 SYNTAX ANALYSIS: {source}")
        print("-" * 70)
        
        try:
            if self.input_mode == 'file':
                with open(self.current_file, 'r') as f:
                    code = f.read()
            else:
                code = self.current_code
                
            lexer = Lexer()
            parser = Parser(lexer.tokenize(code))
            ast = parser.parse()
            
            print("\n📊 Abstract Syntax Tree:")
            print()
            self._print_ast(ast, indent=0)
            
            print(f"\n✅ Syntax analysis completed successfully!")
            
        except Exception as e:
            print(f"❌ Syntax Error: {str(e)}")
            
    def _print_ast(self, node, indent=0):
        """Recursively print AST"""
        prefix = "  " * indent
        
        if hasattr(node, '__class__'):
            print(f"{prefix}• {node.__class__.__name__}")
            
            if hasattr(node, '__dict__'):
                for key, value in node.__dict__.items():
                    if isinstance(value, list):
                        print(f"{prefix}  - {key}: [{len(value)} items]")
                        for item in value:
                            self._print_ast(item, indent + 2)
                    elif hasattr(value, '__class__') and hasattr(value, '__dict__'):
                        print(f"{prefix}  - {key}:")
                        self._print_ast(value, indent + 2)
                    else:
                        print(f"{prefix}  - {key}: {value}")
                        
    def validate_file(self):
        """Validate semantic analysis"""
        if not self.current_file and not self.current_code:
            print("❌ No file or code loaded! Please load a file or write DSL code first.")
            return
            
        source = self.current_file if self.input_mode == 'file' else "Direct Input"
        print(f"\n✅ SEMANTIC VALIDATION: {source}")
        print("-" * 70)
        
        try:
            if self.input_mode == 'file':
                result = self.compiler.compile_file(self.current_file, output_path=None)
            else:
                result = self.compiler.compile_string(self.current_code, output_path=None)
            
            if result['success']:
                print("\n✅ All validations passed!")
                print(f"   • Lexical analysis: OK")
                print(f"   • Syntax analysis: OK")
                print(f"   • Semantic analysis: OK")
                print(f"   • Model: {result['ast'].model.name}")
                print(f"   • Layers: {len(result['ast'].model.layers)}")
            else:
                print("\n❌ Validation failed!")
                for error in result.get('errors', []):
                    print(f"   • {error}")
                    
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
    def compile_file(self):
        """Compile DSL to Python"""
        if not self.current_file and not self.current_code:
            print("❌ No file or code loaded! Please load a file or write DSL code first.")
            return
            
        source = self.current_file if self.input_mode == 'file' else "Direct Input"
        print(f"\n⚙️  COMPILATION: {source}")
        print("-" * 70)
        
        # Ensure output directory exists
        import os
        os.makedirs('output', exist_ok=True)
        
        output_path = input("\nEnter output file name (default: output/compiled.py): ").strip()
        if not output_path:
            output_path = "output/compiled.py"
        elif not output_path.startswith('output/'):
            output_path = f"output/{output_path}"
            
        try:
            if self.input_mode == 'file':
                result = self.compiler.compile_file(self.current_file, output_path)
            else:
                result = self.compiler.compile_string(self.current_code, output_path)
            
            if result['success']:
                print(f"\n✅ Compilation successful!")
                print(f"   📝 Output written to: {output_path}")
                
                # Show file info
                if os.path.exists(output_path):
                    size = os.path.getsize(output_path)
                    with open(output_path, 'r') as f:
                        lines = len(f.readlines())
                    print(f"   📊 Generated: {lines} lines, {size} bytes")
            else:
                print(f"\n❌ Compilation failed!")
                for error in result.get('errors', []):
                    print(f"   • {error}")
                    
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
    def compile_and_run(self):
        """Compile and execute Python code"""
        if not self.current_file and not self.current_code:
            print("❌ No file or code loaded! Please load a file or write DSL code first.")
            return
            
        source = self.current_file if self.input_mode == 'file' else "Direct Input"
        print(f"\n🚀 COMPILE & RUN: {source}")
        print("-" * 70)
        
        # Ensure output directory exists
        import os
        os.makedirs('output', exist_ok=True)
        output_path = "output/temp_output.py"
        
        try:
            # Compile
            print("\n⚙️  Compiling...")
            if self.input_mode == 'file':
                result = self.compiler.compile_file(self.current_file, output_path)
            else:
                result = self.compiler.compile_string(self.current_code, "Direct Input")
                # Write output file
                if result['success']:
                    with open(output_path, 'w') as f:
                        f.write(result['code'])
                    result['output_file'] = output_path
            
            if not result['success']:
                print(f"❌ Compilation failed!")
                for error in result.get('errors', []):
                    print(f"   • {error}")
                return
                
            print(f"✅ Compilation successful!")
            print(f"📝 Output written to: {output_path}")
            
            # Ask for confirmation
            response = input("\n🚀 Run the generated code? (y/n): ").strip().lower()
            if response != 'y':
                print("❌ Execution cancelled.")
                return
                
            # Execute
            print("\n" + "=" * 70)
            print("🚀 EXECUTING GENERATED CODE...")
            print("=" * 70)
            print()
            
            os.system(f"python {output_path}")
            
            print("\n" + "=" * 70)
            print("✅ Execution completed!")
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
    def show_examples(self):
        """Show example DSL files"""
        print("\n📚 EXAMPLE DSL FILES")
        print("-" * 70)
        
        examples_dir = Path(__file__).parent / "examples"
        if not examples_dir.exists():
            print("❌ Examples directory not found!")
            return
            
        examples = sorted(examples_dir.glob("*.sndsl"))
        
        for i, ex in enumerate(examples, 1):
            print(f"\n{i}. {ex.name}")
            print("   " + "-" * 66)
            
            with open(ex, 'r') as f:
                lines = f.readlines()[:10]  # First 10 lines
                
            for line in lines:
                if line.strip():
                    print(f"   {line.rstrip()}")
                    
            total_lines = sum(1 for _ in open(ex))
            if total_lines > 10:
                print(f"   ... ({total_lines - 10} more lines)")
                
        print()
        
    def show_help(self):
        """Show help information"""
        print("\n❓ HELP - SimpleNeural-DSL")
        print("-" * 70)
        print("""
🧠 SimpleNeural-DSL adalah Domain Specific Language untuk konfigurasi
   model Machine Learning tanpa menulis kode Python manual.

📋 CARA PENGGUNAAN:
   1. Load file DSL (.sndsl) atau write code directly
   2. Validate untuk cek error
   3. Compile untuk generate Python code
   4. Run untuk eksekusi training model

🔤 CONTOH DSL CODE:
   
   DATASET load "data.csv" TARGET "price"
   
   MODEL "HousePredictor" {
       LAYER DENSE units: 64 activation: "relu"
       LAYER DROPOUT rate: 0.2
       LAYER DENSE units: 1 activation: "linear"
       
       OPTIMIZER "adam" lr: 0.001
       TRAIN epochs: 100 batch_size: 32
   }

� KAGGLE INTEGRATION:
   • Option 3: Auto-generate DSL from Kaggle dataset (SMART!)
   • Option 4: Download dataset langsung dari Kaggle
   • Butuh Kaggle API credentials (kaggle.com/account)
   • Datasets disimpan di folder 'datasets/'
   
   Popular datasets:
   - uciml/iris (Iris classification)
   - heptapod/titanic (Titanic survival)
   - uciml/breast-cancer-wisconsin-data
   - Dan banyak lagi...

📚 DOKUMENTASI LENGKAP:
   • README.md - Overview
   • QUICKSTART.md - Tutorial
   • docs/08-kaggle-integration.md - Kaggle integration guide
   • docs/ - Full documentation
   
💡 TIPS:
   • Mulai dari example files atau option 3 (auto-generate)
   • Gunakan validation sebelum compile
   • Check error messages untuk debugging
   • Download datasets via option 4 sebelum compile
""")
        
    def run(self):
        """Main application loop"""
        while True:
            self.clear_screen()
            self.print_header()
            
            if self.current_file:
                print(f"📄 Current: {self.current_file} (from file)")
                print()
            elif self.current_code:
                print(f"📝 Current: DSL code ({len(self.current_code.split(chr(10)))} lines, direct input)")
                print()
            
            self.print_menu()
            
            choice = input("Choose option: ").strip().upper()
            
            if choice == '1':
                self.load_file()
            elif choice == '2':
                self.write_dsl_code()
            elif choice == '3':
                self.auto_generate_from_kaggle()
            elif choice == '4':
                self.download_kaggle_dataset()
            elif choice == '5':
                self.view_file()
            elif choice == '6':
                self.show_tokens()
            elif choice == '7':
                self.show_ast()
            elif choice == '8':
                self.validate_file()
            elif choice == '9':
                self.compile_file()
            elif choice == 'A':
                self.compile_and_run()
            elif choice == 'B':
                self.show_examples()
            elif choice == 'C':
                self.show_help()
            elif choice == '0':
                print("\n👋 Thank you for using SimpleNeural-DSL!")
                print("=" * 70)
                break
            else:
                print(f"❌ Invalid option: {choice}")
                
            if choice != '0':
                input("\nPress Enter to continue...")


def main():
    """Entry point"""
    ui = SimpleNeuralUI()
    ui.run()


if __name__ == "__main__":
    main()
