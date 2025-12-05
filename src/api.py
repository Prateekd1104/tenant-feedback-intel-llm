from typing import Optional, List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .analysis import FeedbackAnalyzer
from .config import settings

app = FastAPI(
    title=settings.project_name,
    version="0.1.0",
    description="LLM + RAG powered tenant feedback intelligence API",
)

analyzer = FeedbackAnalyzer()


class FeedbackRequest(BaseModel):
    channel: str = Field(..., description="email | review | chat")
    text: str
    tenant_id: Optional[str] = None
    context_id: Optional[str] = None


class FeedbackResponse(BaseModel):
    channel: str
    tenant_id: Optional[str]
    context_id: Optional[str]
    sentiment: str
    emotion: str
    sentiment_confidence: float
    root_cause: str
    retrieved_docs: List[str]
    auto_response: str


class BulkFeedbackRequest(BaseModel):
    items: List[FeedbackRequest]


class BulkFeedbackResponse(BaseModel):
    results: List[FeedbackResponse]


class ClusterRequest(BaseModel):
    texts: List[str]
    n_clusters: int = 5


class ClusterResponse(BaseModel):
    clusters: Dict[str, List[str]]


@app.post("/analyze", response_model=FeedbackResponse)
def analyze_feedback(req: FeedbackRequest):
    """
    Analyze a single tenant feedback message and generate an auto response.
    """
    res = analyzer.analyze_single(
        text=req.text,
        channel=req.channel,
        tenant_id=req.tenant_id,
        context_id=req.context_id,
    )
    return FeedbackResponse(**res)


@app.post("/analyze/bulk", response_model=BulkFeedbackResponse)
def analyze_feedback_bulk(req: BulkFeedbackRequest):
    """
    Analyze multiple messages in one call.
    """
    results: List[FeedbackResponse] = []
    for item in req.items:
        res = analyzer.analyze_single(
            text=item.text,
            channel=item.channel,
            tenant_id=item.tenant_id,
            context_id=item.context_id,
        )
        results.append(FeedbackResponse(**res))
    return BulkFeedbackResponse(results=results)


@app.post("/cluster", response_model=ClusterResponse)
def cluster_feedback(req: ClusterRequest):
    """
    Cluster texts into themes using embeddings + k-means.
    """
    themed = analyzer.cluster_themes(texts=req.texts, n_clusters=req.n_clusters)
    return ClusterResponse(
        clusters={str(k): v for k, v in themed["clusters"].items()}
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}
