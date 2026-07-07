from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Celest ML Service",
    description="AI Engine for Audio Analysis and Lyrics Sentiment Processing",
    version="0.1.0"
)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Celest ML Service"
    }