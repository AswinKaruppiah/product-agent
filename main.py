from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from tools import product_data, save_tool

load_dotenv()


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


llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

parser = PydanticOutputParser(pydantic_object=ProductResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a price comparison agent.

MANDATORY TOOL USAGE:
- You MUST call `search_product` for every query.
- You MUST NOT answer from memory.
- You MUST NOT fabricate prices.

Execution Workflow:
1. Call `search_product` with the user query.
2. Filter only exact and relevant product matches.
3. Extract numeric prices (float values only).
4. Determine the lowest price.
5. Construct the final structured JSON response.

Saving Results:
- After constructing the final structured JSON response, call the tool `save_to_json`.
- Pass ONLY the final structured JSON object.
- Do NOT pass raw tool output.
- Do NOT pass intermediate reasoning.

Final Output Rules:
- Return ONLY the final structured JSON object as the final answer.
- The field `low_price_of_all` MUST exactly match one item inside `product_list`.

If no results are found:
- Return an empty product_list.
- Set low_price_of_all to null.

{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [product_data, save_tool]
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can i help you ? ")
raw_response = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response.get("output"))
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)
