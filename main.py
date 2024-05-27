from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from model import model

app = FastAPI()

class PromptRequest(BaseModel):
    system_message: str
    instruction: str

class PromptResponse(BaseModel):
    generated_text: str

@app.post("/generate", response_model=PromptResponse)
async def generate(request: PromptRequest):
    prompt = f"<SYS> {request.system_message} <INST> {request.instruction} <RESP> "
    generated_text = model.generate_text(prompt)
    return PromptResponse(generated_text=generated_text)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
