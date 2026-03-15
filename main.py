from fastapi import FastAPI
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from tools import product_data, save_tool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PriceItem(BaseModel):
    website: str
    title: str
    price: float
    currency: str
    product_link: str


class ProductResponse(BaseModel):
    name: str
    product_list: List[PriceItem]
    low_price_of_all: PriceItem

class ChatRequest(BaseModel):
    message: str

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=1,
    top_p=1,
    max_completion_tokens=4096,
    streaming=True
)

parser = PydanticOutputParser(pydantic_object=ProductResponse)
prompt = ChatPromptTemplate.from_messages(
[
(
"system",
"""
You are a STRICT price comparison agent.

MANDATORY TOOL USAGE
- You MUST call `search_product`.
- NEVER answer from memory.
- NEVER fabricate prices.

QUERY INTERPRETATION
First determine the PRIMARY PRODUCT CATEGORY from the user query.

Examples:
- "iqoo" → smartphone
- "iphone 15" → smartphone
- "sony headphones" → headphones
- "samsung tv" → television

PRODUCT MATCH RULES

1. CATEGORY MATCH (VERY IMPORTANT)
   - Only include products from the same category.
   - If the query is a smartphone brand or model, include ONLY smartphones.

2. ACCESSORY EXCLUSION
   ALWAYS exclude accessories such as:
   - cases
   - back cover
   - screen guard
   - charger
   - cable
   - adapter
   - earphones
   - tempered glass
   - battery

3. TITLE VALIDATION
   The product title MUST:
   - contain the main query keyword
   - belong to the same category

4. REJECT PRODUCTS if title contains any of these words:
   case
   cover
   back cover
   pouch
   charger
   cable
   adapter
   tempered
   protector
   skin

5. MODEL MATCH
   If the query specifies a model, include ONLY that model.

RESULT COUNT
- Return at least 10 products if available.

OUTPUT RULES
- Return ONLY the JSON object.
- Do not include explanations.

{format_instructions}
"""
),
("placeholder", "{chat_history}"),
("human", "{input}"),
("placeholder", "{agent_scratchpad}"),
]
).partial(format_instructions=parser.get_format_instructions())

tools = [product_data]
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.get("/")
def home():
    return {"status": "server running"}

@app.post("/chat")
async def chat(req: ChatRequest):
    query = req.message

    raw_response = agent_executor.invoke({"input": query})

    try:
        structured_response = parser.parse(raw_response.get("output"))
        return {"response": structured_response}
    except Exception as e:
        return {
            "error": "Error parsing response",
            "raw_response": raw_response
        }