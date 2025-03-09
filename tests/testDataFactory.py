import os

import nltk
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, DirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator


def test_dataCreation():
    # create object of class of that specific metric
    project_root = os.path.dirname(os.path.abspath(__file__))
    nltk.data.path.append(os.path.join(project_root, "nltk_data"))
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)
    embed = OpenAIEmbeddings()
    loader = DirectoryLoader(
        path="Courses/LLM_Testing/LLMEvaluation_Resources/fs11/",
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader,
    )
    docs = loader.load()
    generate_embeddings = LangchainEmbeddingsWrapper(embed)
    generator = TestsetGenerator(llm=langchain_llm, embedding_model=generate_embeddings)
    dataset = generator.generate_with_langchain_docs(docs, testset_size=20)
    print(dataset.to_list())
    dataset.upload()
