# F1Answers

Formula 1 RAG system providing driver bios, Grand Prix stats, team and track info.
Answers questions and delivers structured F1 insights.

To get started, you need to create a .env file of the form:

```txt
OPENAI_API_KEY=<YOUR API KEY>
EMBEDDINGS_BASE_URL=<YOUR BASE_URL FOR EMBEDDER>
CHATGPT_BASE_URL=<YOUR BASE_URL FOR GENERATOR>
```

After that, you need to build and run the Docker container:

```bash
docker run
```
