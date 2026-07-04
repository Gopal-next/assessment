from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from schema.request import ChatRequest
from tools.retrieval import search_assessments
from agent.shl_agent import invoke
from pathlib import Path

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent.parent

memory = []

@app.get("/")
def home():
    return RedirectResponse("/docs")

@app.get("/health")
def health():
    return {"status": "ok"}

def load_completion_phrases():
    with open(
        BASE_DIR / "data" / "completion_text.txt",
        encoding="utf8"
    ) as f:
        return [

            line.strip().lower()
            for line in f

            if line.strip()

        ]
    
completion_phrases = load_completion_phrases()

@app.post("/chat")
def chat(req: ChatRequest):
    global memory
    for msg in req.messages:
        if msg not in memory:
            memory.append(msg)

    history = "\n".join(
        f"{m.role}: {m.content}"
        for m in memory

    )

    result = invoke(history)

    reply = result["output"]

    if isinstance(reply, list):
        reply = "".join(
            item["text"]
            if isinstance(item, dict) and "text" in item
            else str(item)
            for item in reply
        )

    elif isinstance(reply, dict):
        reply = reply.get(
            "text",
            str(reply)
            )
        
    else:
        reply = str(reply)
    reply = reply.strip()

    recommendations = []

    is_clarification = "?" in reply

    if not is_clarification:

        recs = search_assessments(history)
        recommendations = [
            {
                "name": r["name"],
                "url": r["link"],
                "test_type":
                    r["keys"][0]
                    if r["keys"]
                    else "General"
                }
                for r in recs
                ]

    last_message = req.messages[-1].content.lower().strip()

    end_of_conversation = any(phrase in last_message for phrase in completion_phrases)
    # print(history)
    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": end_of_conversation
        }