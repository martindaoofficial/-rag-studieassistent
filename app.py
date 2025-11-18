# ===========================================
# app.py
# Flask-app til at spørge ind til PDF-materiale for en valgt lektion.
# Nu med filtrering pr. lektion og forbedret fejl-håndtering.
# ===========================================

import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
import google.genai as genai

# -------------------------------------------
# 1. Opsætning
# -------------------------------------------

# Indlæs .env (så GEMINI_API_KEY kan bruges)
load_dotenv()

DB_DIR = "vectorstore"  # Her ligger din Chroma database
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "models/gemini-2.5-flash-preview-09-2025"  # Hurtig og gratis model - se evt. list_models.py

# Initialiser SentenceTransformer

# Dummy embedder-klasse (erstatter SentenceTransformer)
class SimpleEmbedder:
    def embed_documents(self, texts):
        return texts

    def embed_query(self, text):
        return text

embedder = SimpleEmbedder()

# Indlæs Chroma-database
vs = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embedder,
    collection_name="fs_rag"
)

# Opsæt Gemini-klient (kræver GEMINI_API_KEY i .env)
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("⚠️ GEMINI_API_KEY mangler i .env-filen.")
genai_client = genai.Client(api_key=api_key)

# Prompt der styrer svarformat
RAG_PROMPT = PromptTemplate.from_template(
    """
Du er en studieassistent, der SKAL svare i et meget struktureret, overskueligt og let-læseligt format.

📌 **REGLER FOR FORMAT (meget vigtigt):**
- Brug ALWAYS Markdown.
- Start med en H1-overskrift der beskriver svaret.
- Brug H2 til undersektioner.
- Brug punktlister under alle sektioner, og efter hvert punkt indsæt en reference i form af [x].
- Marker nøglebegreber med **fed**.
- Brug korte sætninger.
- Lav ingen lange tekstblokke.
- Brug referencer som [1], [2], osv. i stedet for fodnoter.
- Alle referencer samles i bunden i en *Kilder*-sektion.

📌 **Hvis informationen ikke findes i kilderne**, sig:
> "Oplysningen findes ikke i dine materialer for denne lektion."

📌 **Output struktur (følg 100%):**

# Overskrift (H1)

## 1. Nøglepunkter
- Punkt[1]
- Punkt[2]

## 2. Forklaring
- Kort forklaring

## 3. Eksempler (hvis muligt)
- Eksempel

---

## Kilder
{context_ref_list}

---

## Kildetekst (til modellen – ikke vis til brugeren)
{context_text}

---

Brugerens spørgsmål:
{question}

"""
)

# -------------------------------------------
# 2. Retrieval-funktioner
# -------------------------------------------

def retrieve(query, k=12, filters=None):
    """Finder de mest relevante tekststykker fra databasen."""
    try:
        if filters:
            docs = vs.similarity_search(query, k=k, filter=filters)
        else:
            docs = vs.similarity_search(query, k=k)
        return docs
    except Exception as e:
        print(f"Fejl under retrieval: {e}")
        return []

def format_context(docs):
    """Formatterer referencer OG fuld konteksttekst til prompten."""
    refs = []
    context_texts = []

    for i, d in enumerate(docs, start=1):
        lektion = d.metadata.get("lektion", "Ukendt")
        fil = d.metadata.get("source_file", "?")
        page = d.metadata.get("page", "?")

        refs.append(f"[{i}] {lektion}/{fil}, s.{page}")
        context_texts.append(f"[{i}] {d.page_content}")

    return {
        "ref_list": "\n".join(refs),
        "context_text": "\n\n".join(context_texts)
    }

# -------------------------------------------
# 3. Flask-app
# -------------------------------------------

app = Flask(__name__)

@app.route("/")
def home():
    """Viser startsiden (index.html)"""
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    """Modtager spørgsmål fra brugeren og returnerer svar"""
    q = request.json.get("q", "")
    lektion = request.json.get("lektion", "").strip()

    # Filtrér kun dokumenter fra den valgte lektion
    query_filter = {"lektion": lektion} if lektion else None

    docs = retrieve(q, k=12, filters=query_filter)
    ctx = format_context(docs)
    raw_prompt = RAG_PROMPT.format(
        question=q,
        context_ref_list=ctx["ref_list"],
        context_text=ctx["context_text"]
    )

    # Gør prompten ASCII-sikker (workaround for encoding-fejl i nogle miljøer)
    safe_prompt = raw_prompt.encode("ascii", errors="ignore").decode("ascii", errors="ignore")

    # Send prompt til Gemini og håndtér evt. fejl
    try:
        resp = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=safe_prompt
        )
        answer = getattr(resp, "text", None) or getattr(resp, "output_text", None) or "Fejl: Kunne ikke laese modelsvar."
    except Exception as e:
        answer = f"Fejl ved hentning af svar: {e!r}"

    sources = [
        {
            "lektion": d.metadata.get("lektion"),
            "file": d.metadata.get("source_file"),
            "page": d.metadata.get("page"),
        }
        for d in docs
    ]

    return jsonify({"answer": answer, "sources": sources})

# -------------------------------------------
# 4. Run server
# -------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)