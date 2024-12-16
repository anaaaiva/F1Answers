# F1Answers

Formula 1 RAG system providing driver bios, Grand Prix stats, team and track info.
Answers questions and delivers structured F1 insights.

To get started, you need to create a .env file of the form:

```txt
OPENAI_API_KEY=<YOUR API KEY>
EMBEDDER_BASE_URL=<YOUR BASE_URL FOR EMBEDDER>
GENERATOR_BASE_URL=<YOUR BASE_URL FOR GENERATOR>
```

After that, you need to build and run the Docker container:

```bash
docker build --rm -t f1answers:latest .
docker volume create f1answers_index
docker run --name f1answers_bot -p 8501:8501 -v f1answers_index:/app/index f1answers:latest
```
