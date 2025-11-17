import os
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Sørg for at køre: pip install langchain-text-splitters

# === Opsætning ===
DATA_DIR = os.path.join(os.path.dirname(__file__), "docs") #RAG scanner kun docs-mappen i det projekt, aldrig andet.
DB_DIR = "vectorstore"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
emb_model = SentenceTransformer(EMB_MODEL_NAME)

# === Embedder ===
class SimpleEmbedder:
    def embed_documents(self, texts):
        return emb_model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, text):
        return emb_model.encode([text], normalize_embeddings=True).tolist()[0]

embedder = SimpleEmbedder()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# === Indsæt ALLE PDF’er i alle lektioner og undermapper ===
all_docs = []

for root, dirs, files in os.walk(DATA_DIR):
    # Vi skipper mapper der ikke er lektioner (såsom docs/, .venv/ osv.)
    # lektion_folder hentes direkte fra mappenavnet (fx 'Lektion_1')
    if not os.path.basename(root).startswith("Lektion_"):
        continue
    lektion_folder = os.path.basename(root)
    for file in files:
        if file.endswith(".pdf"):
            path = os.path.join(root, file)
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            for p in pages:
                p.metadata["lektion"] = lektion_folder
                p.metadata["source_file"] = file
            all_docs.extend(pages)


# === Gem i Chroma ===
vs = Chroma.from_documents(
    documents=all_docs,
    embedding=embedder,
    persist_directory=DB_DIR,
    collection_name="fs_rag"
)

# Vi holder styr på unikke lektionsmapper
lektioner = set()
for root, _, files in os.walk(DATA_DIR):
    if os.path.basename(root).startswith("Lektion_"):
        if any(f.endswith(".pdf") for f in files):
            lektioner.add(os.path.basename(root))

print(f"📚 Indlæst {len(all_docs)} dokumentstykker fra {len(lektioner)} lektioner.")