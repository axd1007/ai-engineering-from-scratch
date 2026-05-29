from qdrant_client import QdrantClient
import hashlib
import numpy as np
import json
from datetime import datetime

print("Подключаемся к Qdrant...")
client = QdrantClient(host="qdrant", port=6333, check_compatibility=False)

points = client.scroll(collection_name="news", limit=100)[0]
print(f"Найдено документов: {len(points)}")

categories = {}
for point in points:
    cat = point.payload["category"]
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(point.payload["text"])

def text_to_vector(text, size=128):
    np.random.seed(int(hashlib.md5(text.encode()).hexdigest(), 16) % 2**32)
    return np.random.rand(size).tolist()

query = "machine learning and neural networks"
results = client.query_points(
    collection_name="news",
    query=text_to_vector(query),
    limit=3
).points

report = {
    "timestamp": datetime.now().isoformat(),
    "total_documents": len(points),
    "categories": {cat: len(docs) for cat, docs in categories.items()},
    "search_query": query,
    "top_matches": [
        {"score": round(r.score, 4), "text": r.payload["text"], "category": r.payload["category"]}
        for r in results
    ]
}

output_path = "/results/report.json"
with open(output_path, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\nРезультат сохранён в {output_path}")
print(json.dumps(report, indent=2, ensure_ascii=False))
