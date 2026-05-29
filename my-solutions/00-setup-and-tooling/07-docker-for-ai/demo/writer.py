from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import hashlib
import numpy as np

print("Подключаемся к Qdrant...")
client = QdrantClient(host="qdrant", port=6333)

client.recreate_collection(
    collection_name="news",
    vectors_config=VectorParams(size=128, distance=Distance.COSINE)
)
print("Коллекция создана")

# генерируем данные без скачивания
documents = [
    {"text": "Neural networks learn from data using backpropagation", "category": "sci.ai"},
    {"text": "Deep learning requires large datasets and GPU compute", "category": "sci.ai"},
    {"text": "Transformers use attention mechanisms for NLP tasks", "category": "sci.ai"},
    {"text": "NASA launched a new rocket to the International Space Station", "category": "sci.space"},
    {"text": "Mars rover discovered evidence of ancient water on the planet", "category": "sci.space"},
    {"text": "Hubble telescope captured images of distant galaxies", "category": "sci.space"},
    {"text": "Doctors use MRI scans to detect brain tumors early", "category": "sci.med"},
    {"text": "New vaccine shows 95% efficacy in clinical trials", "category": "sci.med"},
    {"text": "CRISPR gene editing can cure hereditary diseases", "category": "sci.med"},
    {"text": "Quantum computers can solve problems faster than classical ones", "category": "sci.physics"},
]

def text_to_vector(text, size=128):
    np.random.seed(int(hashlib.md5(text.encode()).hexdigest(), 16) % 2**32)
    return np.random.rand(size).tolist()

points = [
    PointStruct(
        id=i,
        vector=text_to_vector(doc["text"]),
        payload={"text": doc["text"], "category": doc["category"]}
    )
    for i, doc in enumerate(documents)
]

client.upsert(collection_name="news", points=points)
print(f"Записано {len(points)} документов")
print(f"Коллекции: {client.get_collections()}")
