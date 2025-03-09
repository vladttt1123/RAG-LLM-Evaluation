import pytest
from ragas import SingleTurnSample, EvaluationDataset, evaluate

from utils import load_test_data, get_llm_response


@pytest.mark.parametrize("getData", load_test_data("Test5.json"), indirect=True)
@pytest.mark.asyncio
async def test_relevancy_factual(llm_wrapper, getData):
    # if metrics are omitted, the deafult mostly used metrics are used
    # metrics = [ResponseRelevancy(llm=llm_wrapper)
    #     , FactualCorrectness(llm=llm_wrapper)]

    # is used when multiple metric are used, convert to ragas_dataset
    eval_dateset = EvaluationDataset([getData])
    # results = evaluate(dataset=eval_dateset, metrics=metrics)
    results = evaluate(dataset=eval_dateset)
    print(results)
    for relevancy_score in results["answer_relevancy"]:
        assert relevancy_score > 0.7
    #
    # for factual_score in results["factual_correctness"]:
    #     assert factual_score > 0.7

    results.upload()


@pytest.fixture
def getData(request):
    test_data = request.param

    responseDict = get_llm_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=responseDict["answer"],
        retrieved_contexts=[doc["page_content"] for doc in responseDict.get("retrieved_docs", [])],
        reference=test_data["reference"]
    )
    return sample
