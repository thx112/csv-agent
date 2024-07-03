from langchain_community.chat_models import ChatOpenAI
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_types import AgentType

from typing import Optional, List
import os

from .csv_base import create_csv_agent

# MAIN_DIR = os.getcwd()

# embedding_model = OpenAIEmbeddings()
# vectorstore = FAISS.load_local(MAIN_DIR / "titanic_data", embedding_model)
# retriever_tool = create_retriever_tool(
#     vectorstore.as_retriever(), "person_name_search", "Search for a person by name"
# )

import access

llm = ChatOpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url=os.environ.get("DASHSCOPE_API"),
    model="qwen2-72b-instruct",
    top_p=0.9,
    temperature=0.1
)


# file_id = "1"  # 从会话中获取文件 ID
memory = ConversationBufferMemory(memory_key="chat_history", 
                                  input_key="input",
                                  output_key="output",
                                  return_messages=True)

prefix = """你的名字叫小美, 由工程师爸爸开发, 可以解决数据问题的人工智能工具。
        与人类进行对话，尽力回答以下问题。您可以使用以下工具：
        开始！{chat_history} 问题：{input}
        """

agent_executor = create_csv_agent(
    llm=llm,
    path= "Floor.csv",
    verbose=True,
    prefix=prefix,
    memory = memory,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
)

class AgentInputs(BaseModel):
    input: str
    user_id: Optional[str] = Field("")
    file_name: Optional[str] = Field("")

agent_executor = agent_executor.with_types(input_type=AgentInputs)
