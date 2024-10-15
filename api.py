from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
import json, uvicorn
from fastapi import FastAPI, Response
from pydantic import BaseModel
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex

system_prompt_llama = """Your responses should be precise and exclusively related to female reproductive health. 
You always bold the response that are important, title, heading, label, bullet points or points. 
You always give related links keep bolded. 
Refrain from addressing queries related to programming, other banks, company, organization or any unrelated subjects other than female reproductive health. 
Ensure the authenticity and relevance of provided links and email address to female reproductive health resources. 
In instances of inquiries outside our scope, politely respond with I do not have information on this topic. 
Focus your expertise on the female reproductive health. 
Limit your responses to the inquiries posed, disregarding questions not related to female reproductive health. 
You do not know anything other than female reproductive health. 
"""

inp_urls = [
            'https://bmcwomenshealth.biomedcentral.com/articles', 
            'https://indianmenopausesociety.org/info-for-professionals.html', 
            'https://vikaspedia.in/health/women-health', 
            ]

llm = Ollama(model="llama3.2", request_timeout=60.0)
# data = SimpleDirectoryReader(input_dir="sbichatbot/data_pdf/").load_data(show_progress = True)
data = SimpleWebPageReader(html_to_text=True).load_data(inp_urls)
Settings.llm = llm
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900

index_llama = VectorStoreIndex.from_documents(data, embed_model=embed_model)


sess = {}

app = FastAPI()

class Input(BaseModel):
    question: str
    sess_id: int

@app.post("/chat")
def chat(input: Input):
    question = input.question
    sess_id = input.sess_id

    memory = ChatMemoryBuffer.from_defaults(token_limit=100000)

    if sess_id not in sess.keys():
        sess[sess_id] = index_llama.as_chat_engine(chat_mode='context', memory=memory, system_prompt=system_prompt_llama)

    resp = sess[sess_id].chat(f'{question}')
    score = max([x.score for x in resp.source_nodes])
    result = resp.response
    result = result + f'\n \n **Score : {score}%**'
    response_ = {
        "result": result
        }
    
    return Response(content=json.dumps(response_, default=str), headers={"Content-Type": "application/json"})

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=2048, workers=1, reload=False)