from datasets import load_dataset
import os

print("Загружаем датасет...")
dataset = load_dataset("stanfordnlp/imdb", split="train[:1000]")

os.makedirs("data", exist_ok=True)

# конвертируем в разные форматы
dataset.to_csv("data/imdb_train.csv")
dataset.to_json("data/imdb_train.json")
dataset.to_parquet("data/imdb_train.parquet")

# сравниваем размеры
for fmt in ["csv", "json", "parquet"]:
    size = os.path.getsize(f"data/imdb_train.{fmt}")
    print(f"{fmt.upper()}: {size / 1024 / 1024:.2f} MB")

# сплиты
print("\n--- Data Splits ---")
dataset = load_dataset("stanfordnlp/imdb", split="train")

split = dataset.train_test_split(test_size=0.2, seed=42)
train_val = split["train"].train_test_split(test_size=0.125, seed=42)

train_ds = train_val["train"]
val_ds = train_val["test"]
test_ds = split["test"]

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
print(f"Total: {len(train_ds) + len(val_ds) + len(test_ds)}")

# скачивание модели
print("\n--- Model Download ---")
from huggingface_hub import hf_hub_download, snapshot_download

model_path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="config.json"
)
print(f"Config cached at: {model_path}")

model_dir = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
print(f"Full model at: {model_dir}")
