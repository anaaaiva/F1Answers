# F1Answers

Formula 1 RAG system providing driver bios, Grand Prix stats, team and track info.
Answers questions and delivers structured F1 insights.

To get started, you need to create a .env file of the form:

```txt
OPENAI_API_KEY=<YOUR API KEY>
EMBEDDER_BASE_URL=https://gptunnel.ru/v1/embeddings
GENERATOR_BASE_URL=https://gptunnel.ru/v1/chat/completions
```

Then you need to build and run the Docker container:

```bash
docker-compose up --build
```

After that, it can be run again without build:

```bash
docker-compose up
```

And to stop and remove the container:

```bash
docker-compose down --volumes
```
