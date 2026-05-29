# Docker Demo: Two-Container Test

## Что делали
- writer.py — генерирует данные и пишет в Qdrant
- reader.py — читает из Qdrant, ищет похожие, сохраняет отчёт на хост

## Результат
- 10 документов записано в коллекцию "news"
- Семантический поиск работает через vector similarity
- Файл report.json сохранён на хосте через volume mount

## Команды
```bash
docker compose up -d qdrant
docker run --rm --network 07-docker-for-ai_default \
    -v ./demo:/workspace ai-dev bash -c \
    "pip install qdrant-client -q && python /workspace/writer.py"
docker run --rm --network 07-docker-for-ai_default \
    -v ./demo:/workspace -v ~/results:/results ai-dev bash -c \
    "pip install qdrant-client -q && python /workspace/reader.py"
```
