import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
load_dotenv()
@pytest.fixture
def llm_wrapper():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)
    return langchain_llm
