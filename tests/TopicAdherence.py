import pytest
from langchain.agents.chat.prompt import HUMAN_MESSAGE
from langchain_core.messages import AIMessage
from ragas import MultiTurnSample
from ragas.metrics import TopicAdherenceScore

from utils import load_test_data, get_llm_response


@pytest.mark.parametrize("getData", load_test_data("Test4.json"), indirect=True)
@pytest.mark.asyncio
async def test_topicAdherence(llm_wrapper, getData):
    topicScore = TopicAdherenceScore(llm=llm_wrapper)
    score = await topicScore.multi_turn_ascore(getData)
    assert score > 0.8


@pytest.fixture
def getData(request):
    test_data = request.param

    responseDict = get_llm_response(test_data)
    conversation = [
        HUMAN_MESSAGE(content="how many articles are there n selenium course?"),
        AIMessage(content="there are 23 articles in the course"),
        HUMAN_MESSAGE(content="how many downloadable resources are there in selenium course?"),
        AIMessage(content="there are 9 downloadable resources"),
    ]

    reference = ["""The AI should: 
    1 Give results related to the selenium webdriver pythongcourse
    2. There are 23 articles and 9 downloadable resources in the course
    """]

    sample = MultiTurnSample(user_input=conversation, reference_topics=reference)

    return sample
