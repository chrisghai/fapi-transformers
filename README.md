# fapi-transformers

The project should have a `.env` file in its root. An example of how it can look like is:

```
PROJECT_NAME=fapi-transformers
BACKEND_CORS_ORIGINS=["http://localhost:8000", "https://localhost:8000", "http://localhost", "https://localhost"]

NLP_ROOT="/opt/fapi-transformers/models"
QA_MODEL={"path": "${NLP_ROOT}/path-to-qa-model", "load": true}
ZS_MODEL={"path": "${NLP_ROOT}/path-to-zero-shot-model", "load": true}
NER_MODEL={"path": "${NLP_ROOT}/path-to-ner-model", "load": true}

POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_SERVER=database
POSTGRES_DB=app
```

## Models

Models compatible with the `transformers` library (e.g. models from [https://huggingface.co/models](https://huggingface.co/models)) should be placed either in a `models/` folder from root of the project, or specified from somewhere else by changing `services.app.volumes` in `docker-compose.yaml`.

## Running

Should be run with `docker-compose`:

`docker-compose up --build -d`

Afterwards, API documentation will be available at [http://localhost:8000/docs#/](http://localhost:8000/docs#/).

## License

This project is licensed under the terms of the MIT license.
