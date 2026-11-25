# German to English Translator

A German to English neural machine translation system. It is a sequence-to-sequence model with LSTMs, trained on the Multi30k dataset, then exported to ONNX and quantised so it runs fast, and served through a FastAPI API.

## What is in this repo

| Path | Purpose |
|---|---|
| `notebook/train.ipynb` | Training notebook. Loads `bentrevett/multi30k`, tokenises with spaCy (`de_core_news_sm` and `en_core_web_sm`), builds the vocabularies, trains the model and tracks experiments with MLflow. |
| `architecture.py` | The model: a bidirectional LSTM `Encoder`, an LSTM `Decoder` and a `Seq2Seq` wrapper, plus translation functions for PyTorch and ONNX. |
| `models/` | Trained PyTorch weights (`best.pt`), ONNX encoder and decoder (`encoder.onnx`, `decoder.onnx`), quantised versions (`*_quant.*`) and the vocabularies (`en_vocab.json`, `de_vocab.json`). |
| `quantscripts/` | `quantizeonnx.py` (dynamic quantisation with ONNX Runtime) and `quantizetorch.py` (PyTorch quantisation). |
| `benchmark.py` | Compares the PyTorch and ONNX Runtime models. |
| `main.py` | FastAPI app that translates with the ONNX models (maximum length 25 tokens). |
| `Dockerfile` | Container image running the API with uvicorn on port 8000. |

## API

| Route | Description |
|---|---|
| `GET /` | Basic root route |
| `POST /translate` | Body `{"text": "..."}`. Returns `{"translated_text": "...", "original_text": "..."}` |

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Ein Mann geht die Straße entlang."}'
```

## Run

Requires Python 3.11 or newer.

```bash
pip install -r requirements.txt          # includes the spaCy English and German models
uvicorn main:app --port 8000
```

The `Dockerfile` copies `mainonnx.py` and starts `mainonnx:app`, but the API file in this repo is named `main.py`. Update the Dockerfile to use `main.py` and `main:app` before building the image.
