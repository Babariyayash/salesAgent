import os
import certifi
from dotenv import load_dotenv

# Ensure the correct SSL certificates are used
os.environ['SSL_CERT_FILE'] = certifi.where()

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
