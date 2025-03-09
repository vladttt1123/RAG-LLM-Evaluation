import pytest
import requests
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference


@pytest.mark.asyncio
async def test_context_precision():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # converting llm object in an object that ragas can understand ( since llm comes from langchain framwork and precision comes from ragas framework)
    langchain_llm = LangchainLLMWrapper(llm)

    context_precision = LLMContextPrecisionWithoutReference(llm=langchain_llm)
    question = "how many articles are there n selenium course?"

    # requests
    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask",
                                 json={
                                     "question": question,
                                     "chat_history": []
                                 }).json()
    print(responseDict)

    # feed data
    sample = SingleTurnSample(
        user_input=question,
        response=responseDict["answer"],
        retrieved_contexts=[
            responseDict["retrieved_docs"][0]["page_content"],
            responseDict["retrieved_docs"][1]["page_content"]
        ]
    )

    # get the scope
    score = await context_precision.single_turn_ascore(sample)
    print(score)
    assert score > 0.8
