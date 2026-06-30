from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="AI Research Paper Analyzer",
    description="Upload research papers and get AI-powered analysis",
    version="1.0.0"
)

# This allows your frontend (Person 2) to talk to your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "AI Research Analyzer API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}