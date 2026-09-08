# Asistente de Trámites — Universidad del Cauca

Sistema de consulta en lenguaje natural sobre los trámites académicos y
administrativos descritos en `data/dataset2.json` (24 trámites, estructura
anidada con `requisitos`, `procedimiento`, `documentos_requeridos`, FAQs, etc.).

## Por qué RAG + fine-tuning y no un transformer entrenado desde cero

Con 24 trámites y ~280 pares de entrenamiento derivables, **no es viable
entrenar un transformer desde cero**: estos modelos necesitan típicamente
millones de oraciones para aprender la estructura del lenguaje. Entrenar
desde cero con este volumen produce un modelo que memoriza el corpus sin
generalizar a preguntas formuladas de otra manera (overfitting severo) y que
ni siquiera domina la gramática básica del español.

En su lugar, este proyecto usa dos técnicas que sí son apropiadas para este
tamaño de datos:

1. **Fine-tuning** (no entrenamiento desde cero) de un encoder de oraciones
   ya preentrenado en varios idiomas (`paraphrase-multilingual-MiniLM-L12-v2`).
   Solo se ajustan sus pesos para acercar semánticamente las preguntas del
   dominio ("¿qué documentos necesito para transferencia?") a los textos
   correctos del corpus. Esto sí es apropiado con cientos de pares, porque
   el modelo ya sabe español y solo aprende el vocabulario/asociaciones
   específicas de "trámites universitarios del Cauca".

2. **RAG (Retrieval-Augmented Generation)**: la respuesta final nunca es
   "inventada" por una red neuronal generativa libre. Se recuperan los
   fragmentos del reglamento más relevantes mediante similitud semántica
   (FAISS) y la respuesta se construye a partir de ese texto real —
   respetando el requisito de que "responda de manera correcta basado en
   el corpus", evitando alucinaciones.

## Arquitectura

```
dataset2.json
     │
     ▼
preprocess.py ──► chunks.json (236 fragmentos indexables)
     │             train_pairs.json (279 pares pregunta→respuesta)
     ▼
train_encoder.py ──► fine-tunea el bi-encoder con
                      MultipleNegativesRankingLoss
     │
     ▼
build_index.py ──► embebe los chunks con el encoder fine-tuneado
                    y construye un índice FAISS (similitud coseno)
     │
     ▼
rag_chat.py ──► por cada consulta:
                 1. recupera los k chunks más similares
                 2. si coincide muy bien con una FAQ real -> la devuelve tal cual
                 3. si no, sintetiza la respuesta con los campos recuperados
                    (requisitos, procedimiento, documentos...) citando
                    trámite y base normativa
                 4. si no hay nada suficientemente relevante -> lo dice
                    explícitamente en vez de inventar
```

## Chunking: por qué no un solo vector por trámite

Cada trámite se descompone en varios "chunks":
- un **resumen** completo del trámite (para preguntas generales),
- un chunk **por campo relevante** (requisitos, procedimiento, documentos,
  plazos, costos, sanciones, entidad responsable, quién puede solicitarlo),
- un chunk **por cada FAQ** ya redactada en el dataset (la fuente más
  confiable de todas, porque es una respuesta ya validada).

Esto mejora mucho la precisión de recuperación frente a preguntas puntuales
("¿qué documentos necesito para X?") en vez de mezclar todo el trámite en
un solo vector genérico.

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate        # en Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Nota: la primera vez que corras `train_encoder.py` o `build_index.py`,
`sentence-transformers` descargará el modelo base desde Hugging Face
(~470 MB), así que necesitas conexión a internet normal (no restringida)
la primera vez. Después queda cacheado localmente.

## Uso

Opción rápida — todo el pipeline en orden:
```bash
bash run_all.sh
```

O paso a paso:
```bash
python3 src/preprocess.py       # genera chunks.json y train_pairs.json
python3 src/train_encoder.py    # fine-tunea el encoder (unos minutos en CPU)
python3 src/build_index.py      # construye el índice FAISS
python3 src/rag_chat.py         # chat interactivo por consola
```

Ejemplo de sesión:
```
Tú: ¿Qué documentos necesito para cancelar el semestre?
Asistente: [respuesta basada en el campo documentos_requeridos del
trámite correspondiente, citando trámite y base normativa]
```

## Estructura del proyecto

```
tramites_rag/
├── data/
│   ├── dataset2.json       (dataset original)
│   ├── chunks.json         (generado por preprocess.py)
│   └── train_pairs.json    (generado por preprocess.py)
├── src/
│   ├── preprocess.py
│   ├── train_encoder.py
│   ├── build_index.py
│   └── rag_chat.py
├── models/
│   └── tramites-encoder/   (generado por train_encoder.py)
├── index/
│   ├── tramites.faiss      (generado por build_index.py)
│   └── chunks_meta.json
├── requirements.txt
└── run_all.sh
```

## Cómo ampliar esto a futuro

- **Más datos**: si el reglamento crece o se agregan más FAQs reales, el
  fine-tuning mejora directamente (más pares de entrenamiento = mejor
  discriminación semántica).
- **Generación más fluida**: se puede añadir un LLM pequeño local (ej. un
  modelo instructivo en español de pocos parámetros) para redactar la
  respuesta final en lenguaje más natural a partir de los chunks
  recuperados, en vez de la síntesis por plantilla actual — manteniendo
  siempre el contexto recuperado como única fuente de verdad para evitar
  alucinaciones.
- **Interfaz web**: envolver `TramitesRAG` (clase en `rag_chat.py`) en una
  API con FastAPI/Flask para exponerla como chatbot web.
- **Evaluación**: usar las FAQs como conjunto de prueba (retrieval@k,
  MRR) para medir objetivamente si el fine-tuning mejoró la recuperación
  frente al modelo base.
