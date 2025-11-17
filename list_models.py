# Kør et lille script for at se dine genAI modeller
import os
import google.genai as genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

print("🔍 Tilgængelige modeller:")
for m in client.models.list():
    print("-", m.name)