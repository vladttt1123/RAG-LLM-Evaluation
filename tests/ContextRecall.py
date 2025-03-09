import pytest
import requests
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall

@pytest.mark.asyncio
async def test_context_recall():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    langchain_llm = LangchainLLMWrapper(llm)

    context_recall = LLMContextRecall(llm=langchain_llm)

    question = "how many articles are there n selenium course?"
    # requests
    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask",
                                 json={
                                     "question": question,
                                     "chat_history": []
                                 }).json()

    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=[
            responseDict["retrieved_docs"][0]["page_content"],
            responseDict["retrieved_docs"][1]["page_content"]],
        reference="23"

    )
    score = await context_recall.single_turn_ascore(sample)
    print(score)
    assert score > 0.7
