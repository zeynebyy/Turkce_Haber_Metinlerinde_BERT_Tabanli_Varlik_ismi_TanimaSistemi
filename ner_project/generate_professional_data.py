from config import Config
from src.preprocessing.data_generator import AdvancedDataGenerator, save_conll
import random
import os
from pathlib import Path

def main():
    config = Config()
    generator = AdvancedDataGenerator(config)
    
    print(f"Generatig {config.NUM_SYNTHETIC_SAMPLES} samples...")
    data = generator.run(count=config.NUM_SYNTHETIC_SAMPLES)
    
    random.shuffle(data)
    
    train_end = int(len(data) * 0.8)
    val_end = int(len(data) * 0.9)
    
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    
    os.makedirs(config.DATA_PROCESSED, exist_ok=True)
    
    save_conll(Path(config.DATA_PROCESSED) / "train.txt", train)
    save_conll(Path(config.DATA_PROCESSED) / "dev.txt", val)
    save_conll(Path(config.DATA_PROCESSED) / "test.txt", test)
    
    print("Dataset generated and saved to", config.DATA_PROCESSED)

if __name__ == "__main__":
    main()
