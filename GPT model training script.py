import os
import torch
import json
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    GPT2Config,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

class GPT2Trainer:
    def __init__(self, dataset_path="dataset", model_name="gpt2", max_length=512):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        
    def load_text_data(self):
        """Carga todos los archivos de texto del dataset"""
        print("Cargando archivos de texto...")
        
        all_texts = []
        file_count = 0
        
        # Buscar todos los archivos .txt en la carpeta dataset
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content:  # Solo agregar si no está vacío
                                all_texts.append(content)
                                file_count += 1
                                
                        if file_count % 100 == 0:
                            print(f"Procesados {file_count} archivos...")
                            
                    except Exception as e:
                        print(f"Error leyendo {file_path}: {e}")
        
        print(f"Total de archivos cargados: {file_count}")
        print(f"Total de textos: {len(all_texts)}")
        
        if len(all_texts) == 0:
            raise ValueError("No se encontraron archivos de texto en el dataset")
        
        return all_texts
    
    def setup_tokenizer_and_model(self):
        """Configura el tokenizer y modelo GPT-2"""
        print("Configurando tokenizer y modelo GPT-2...")
        
        # Cargar tokenizer pre-entrenado
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        
        # Agregar pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configuración del modelo
        config = GPT2Config.from_pretrained(self.model_name)
        
        # Crear modelo
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name, config=config)
        
        # Redimensionar embeddings si es necesario
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        print(f"Modelo cargado: {self.model.num_parameters():,} parámetros")
        
    def prepare_datasets(self, texts, train_split=0.9):
        """Prepara los datasets de entrenamiento y validación"""
        print("Preparando datasets...")
        
        # Dividir en entrenamiento y validación
        split_idx = int(len(texts) * train_split)
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        
        print(f"Textos de entrenamiento: {len(train_texts)}")
        print(f"Textos de validación: {len(val_texts)}")
        
        # Tokenizar textos
        def tokenize_function(examples):
            # Tokenizar y truncar/rellenar
            tokenized = self.tokenizer(
                examples['text'], 
                truncation=True, 
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Para language modeling, labels = input_ids
            tokenized['labels'] = tokenized['input_ids'].clone()
            
            return tokenized
        
        # Crear datasets
        train_dataset = Dataset.from_dict({'text': train_texts})
        val_dataset = Dataset.from_dict({'text': val_texts})
        
        # Aplicar tokenización
        self.train_dataset = train_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=['text']
        )
        
        self.val_dataset = val_dataset.map(
            tokenize_function,
            batched=True, 
            remove_columns=['text']
        )
        
        print("Datasets preparados exitosamente")
    
    def train_model(self, output_dir="./gpt2_trained", epochs=3, batch_size=4, learning_rate=5e-5):
        """Entrena el modelo GPT-2"""
        print(f"Iniciando entrenamiento...")
        print(f"Épocas: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        
        # Configurar argumentos de entrenamiento
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_steps=500,
            save_steps=1000,
            evaluation_strategy="steps",
            save_strategy="steps",
            logging_steps=100,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_dir='./logs',
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            prediction_loss_only=True,
            gradient_accumulation_steps=2,
            dataloader_pin_memory=True,
            fp16=torch.cuda.is_available(),  # Usar precisión mixta si hay GPU
        )
        
        # Data collator para language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # No es masked language modeling
        )
        
        # Crear trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )
        
        # Entrenar
        print("Comenzando entrenamiento...")
        trainer.train()
        
        # Guardar modelo final
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Modelo guardado en: {output_dir}")
        
        return trainer
    
    def save_model_bin(self, output_dir="./gpt2_trained"):
        """Guarda el modelo en formato .bin"""
        print("Guardando modelo en formato .bin...")
        
        # Guardar estado del modelo
        model_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)
        
        # Guardar configuración
        config_path = os.path.join(output_dir, "config.json")
        self.model.config.to_json_file(config_path)
        
        # Guardar tokenizer
        tokenizer_path = os.path.join(output_dir, "tokenizer.json")
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Archivos guardados:")
        print(f"  - {model_path}")
        print(f"  - {config_path}")
        print(f"  - Tokenizer files en {output_dir}")
        
        return model_path
    
    def plot_training_metrics(self, trainer):
        """Grafica las métricas de entrenamiento"""
        try:
            # Obtener logs de entrenamiento
            logs = trainer.state.log_history
            
            train_losses = []
            eval_losses = []
            steps = []
            eval_steps = []
            
            for log in logs:
                if 'loss' in log:
                    train_losses.append(log['loss'])
                    steps.append(log['step'])
                if 'eval_loss' in log:
                    eval_losses.append(log['eval_loss'])
                    eval_steps.append(log['step'])
            
            # Crear gráficas
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(steps, train_losses, label='Training Loss')
            plt.plot(eval_steps, eval_losses, label='Validation Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            if len(eval_losses) > 1:
                perplexity = [np.exp(loss) for loss in eval_losses]
                plt.plot(eval_steps, perplexity, label='Validation Perplexity')
                plt.xlabel('Steps')
                plt.ylabel('Perplexity')
                plt.title('Validation Perplexity')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('training_metrics.png')
            plt.show()
            
        except Exception as e:
            print(f"Error graficando métricas: {e}")


class GPT2Generator:
    """Clase para usar el modelo entrenado para generar texto"""
    
    def __init__(self, model_path="./gpt2_trained"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Carga el modelo entrenado"""
        try:
            print(f"Cargando modelo desde: {self.model_path}")
            
            # Cargar tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            
            # Cargar modelo
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
            
            # Configurar para inferencia
            self.model.eval()
            
            # Usar GPU si está disponible
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print("Modelo cargado en GPU")
            else:
                print("Modelo cargado en CPU")
                
            return True
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False
    
    def generate_text(self, prompt="", max_length=100, temperature=0.8, num_return_sequences=1):
        """Genera texto usando el modelo entrenado"""
        if not self.model:
            if not self.load_model():
                return None
        
        # Tokenizar prompt
        if prompt:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        else:
            # Si no hay prompt, empezar con token BOS
            input_ids = torch.tensor([[self.tokenizer.bos_token_id or self.tokenizer.eos_token_id]])
        
        # Mover a GPU si está disponible
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        # Generar texto
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        # Decodificar resultados
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def interactive_generation(self):
        """Modo interactivo para generar texto"""
        if not self.load_model():
            return
        
        print("\n" + "="*50)
        print("GENERADOR DE TEXTO GPT-2 - MODO INTERACTIVO")
        print("="*50)
        print("Escribe un prompt y presiona Enter (o 'quit' para salir)")
        print("Comandos especiales:")
        print("  'quit' - Salir")
        print("  'random' - Generar texto aleatorio")
        print("-" * 50)
        
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                
                if prompt.lower() == 'quit':
                    break
                elif prompt.lower() == 'random':
                    prompt = ""
                
                print("\nGenerando...")
                texts = self.generate_text(prompt, max_length=150, num_return_sequences=1)
                
                print("\n" + "-" * 50)
                print("TEXTO GENERADO:")
                print("-" * 50)
                for i, text in enumerate(texts, 1):
                    print(f"{i}. {text}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("¡Hasta luego!")


def train_gpt2():
    """Función principal para entrenar GPT-2"""
    # Verificar que existe la carpeta dataset
    if not os.path.exists('dataset'):
        print("Error: No se encontró la carpeta 'dataset' en el directorio actual.")
        print("Crea una carpeta 'dataset' con archivos .txt")
        return
    
    # Verificar que hay archivos de texto
    txt_files = []
    for root, dirs, files in os.walk('dataset'):
        for file in files:
            if file.lower().endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    
    if not txt_files:
        print("Error: No se encontraron archivos .txt en la carpeta dataset")
        return
    
    print(f"Encontrados {len(txt_files)} archivos de texto")
    
    # Crear trainer
    trainer = GPT2Trainer()
    
    # Cargar datos
    texts = trainer.load_text_data()
    
    # Configurar modelo y tokenizer
    trainer.setup_tokenizer_and_model()
    
    # Preparar datasets
    trainer.prepare_datasets(texts)
    
    # Entrenar
    print(f"\n{'='*50}")
    print("INICIANDO ENTRENAMIENTO GPT-2")
    print(f"{'='*50}")
    
    trained_model = trainer.train_model()
    
    # Guardar en formato .bin
    bin_path = trainer.save_model_bin()
    
    # Graficar métricas
    trainer.plot_training_metrics(trained_model)
    
    print(f"\n{'='*50}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*50}")
    print("Archivos generados:")
    print("  - ./gpt2_trained/pytorch_model.bin")
    print("  - ./gpt2_trained/config.json")
    print("  - ./gpt2_trained/tokenizer.json")
    print("  - ./gpt2_trained/vocab.json")
    print("  - training_metrics.png")
    
    return trainer

    def setup_tokenizer_and_model(self):
        """Configura el tokenizer y modelo GPT-2"""
        print("Configurando tokenizer y modelo GPT-2...")
        
        # Cargar tokenizer pre-entrenado
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        
        # Agregar pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configuración del modelo
        config = GPT2Config.from_pretrained(self.model_name)
        
        # Crear modelo
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name, config=config)
        
        # Redimensionar embeddings si es necesario
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        print(f"Modelo cargado: {self.model.num_parameters():,} parámetros")
        
    def prepare_datasets(self, texts, train_split=0.9):
        """Prepara los datasets de entrenamiento y validación"""
        print("Preparando datasets...")
        
        # Dividir en entrenamiento y validación
        split_idx = int(len(texts) * train_split)
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        
        print(f"Textos de entrenamiento: {len(train_texts)}")
        print(f"Textos de validación: {len(val_texts)}")
        
        # Tokenizar textos
        def tokenize_function(examples):
            # Tokenizar y truncar/rellenar
            tokenized = self.tokenizer(
                examples['text'], 
                truncation=True, 
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Para language modeling, labels = input_ids
            tokenized['labels'] = tokenized['input_ids'].clone()
            
            return tokenized
        
        # Crear datasets
        train_dataset = Dataset.from_dict({'text': train_texts})
        val_dataset = Dataset.from_dict({'text': val_texts})
        
        # Aplicar tokenización
        self.train_dataset = train_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=['text']
        )
        
        self.val_dataset = val_dataset.map(
            tokenize_function,
            batched=True, 
            remove_columns=['text']
        )
        
        print("Datasets preparados exitosamente")


def test_generation(prompt=""):
    """Función para probar la generación de texto"""
    generator = GPT2Generator()
    
    if not generator.load_model():
        print("No se pudo cargar el modelo. ¿Ya entrenaste el modelo?")
        return
    
    print(f"\nGenerando texto con prompt: '{prompt}'")
    texts = generator.generate_text(prompt, max_length=200, num_return_sequences=2)
    
    print("\nTextos generados:")
    print("=" * 50)
    for i, text in enumerate(texts, 1):
        print(f"\n{i}. {text}")
        print("-" * 50)


def interactive_mode():
    """Modo interactivo para generar texto"""
    generator = GPT2Generator()
    generator.interactive_generation()


def check_dataset():
    """Verifica el contenido del dataset"""
    if not os.path.exists('dataset'):
        print("No se encontró la carpeta 'dataset'")
        return False
    
    txt_count = 0
    total_chars = 0
    
    for root, dirs, files in os.walk('dataset'):
        for file in files:
            if file.lower().endswith('.txt'):
                txt_count += 1
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        total_chars += len(content)
                except:
                    pass
    
    print(f"Dataset verificado:")
    print(f"   - Archivos .txt: {txt_count}")
    print(f"   - Total caracteres: {total_chars:,}")
    print(f"   - Promedio por archivo: {total_chars//max(txt_count,1):,} caracteres")
    
    if txt_count == 0:
        print("No se encontraron archivos .txt en el dataset")
        return False
    
    if total_chars < 10000:
        print("⚠️  Advertencia: Dataset pequeño, considera agregar más texto")
    
    return True


def show_usage():
    """Muestra cómo usar el script"""
    print("\n" + "="*60)
    print("ENTRENAMIENTO GPT-2 117M")
    print("="*60)
    print("ESTRUCTURA REQUERIDA:")
    print("├── training.py")
    print("└── dataset/")
    print("    ├── archivo1.txt")
    print("    ├── archivo2.txt")
    print("    └── carpeta/")
    print("        └── archivo3.txt")
    print()
    print("FUNCIONES DISPONIBLES:")
    print("1. check_dataset()      - Verificar dataset")
    print("2. train_gpt2()         - Entrenar modelo")
    print("3. test_generation()    - Probar generación")
    print("4. interactive_mode()   - Modo interactivo")
    print()
    print("FLUJO RECOMENDADO:")
    print("1. check_dataset()")
    print("2. train_gpt2()")
    print("3. test_generation('Tu prompt aquí')")
    print("4. interactive_mode()")
    print()
    print("ARCHIVOS GENERADOS:")
    print("- ./gpt2_trained/pytorch_model.bin  ← Modelo principal")
    print("- ./gpt2_trained/config.json")
    print("- ./gpt2_trained/tokenizer.json")
    print("- training_metrics.png")
    print("="*60)


if __name__ == "__main__":
    # Verificar si PyTorch tiene GPU disponible
    if torch.cuda.is_available():
        print(f"GPU disponible: {torch.cuda.get_device_name()}")
        print(f"   Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Usando CPU para entrenamiento")
    
    show_usage()
    
    # Verificar dataset automáticamente
    check_dataset()
    
    # Descomenta para entrenar automáticamente:
    # train_gpt2()
    
    # Ejemplos de uso después del entrenamiento:
    # test_generation("El clima hoy está")
    # interactive_mode()
