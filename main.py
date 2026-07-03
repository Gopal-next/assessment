from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from schema.request import ChatRequest
from tools.retrieval import search_assessments
from agent.shl_agent import invoke
app = FastAPI()

memory = []
@app.get("/health")
def health():
    db_ok = False  # example check

    if not db_ok:
        raise HTTPException(
            status_code=503,
            detail="Service unhealthy: database not reachable"
        )

    return {"status": "ok"}


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
    print("History :",history)
    if isinstance(reply, list):

        reply = "".join(

            item["text"]
            if isinstance(item, dict) and "text" in item
            else str(item)

            for item in reply

        )

    elif isinstance(reply, dict):

        reply = reply.get("text", str(reply))

    else:

        reply = str(reply)

    reply = reply.strip()

    is_clarification = "?" in reply

    if is_clarification:
        recommendations = []
    else:
        recs = search_assessments(history)

        recommendations = [
            {
                "name": r["name"],
                "url": r["link"],
                "test_type": r["keys"][0] if r["keys"] else "General"
            }
            for r in recs
        ]

    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": False
    }