from fastapi import FastAPI

app = FastAPI(
    title="Quran Insights API",
    description="An API for Quran Insights",
)


@app.get("/")
def read_root():
    return {"status": "online", "message": "Welcome to Quran Insights API"}


@app.get("/health")
def health():
    return {"status": "healthy", "message": "Quran Insights API is running smoothly"}
