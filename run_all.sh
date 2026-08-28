#!/usr/bin/env bash
# Ejecuta el pipeline completo: preprocesar -> fine-tunear encoder -> indexar -> chat
set -e
cd "$(dirname "$0")"

echo "== 1/4 Preprocesando dataset =="
python3 src/preprocess.py

echo "== 2/4 Fine-tuneando el encoder (puede tardar unos minutos en CPU) =="
python3 src/train_encoder.py

echo "== 3/4 Construyendo índice FAISS =="
python3 src/build_index.py

echo "== 4/4 Iniciando chat =="
python3 src/rag_chat.py
