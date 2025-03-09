# LLM Evaluation Project

This project is part of the **Udemy course**: [RAG LLM Evaluation AI Test](https://www.udemy.com/course/rag-llm-evaluation-ai-test/).  
It focuses on evaluating various aspects of **Language Learning Models (LLMs)** using different metrics.

---

## 📂 Project Structure

- **`conftest.py`** – Handles setup for loading environment variables and defining test fixtures.  
- **`pytest.ini`** – Configuration file for Pytest settings.  
- **`tests/`** – Directory containing test files for different evaluation metrics:  
  - 📌 **`ContextPrecision.py`** – Tests the **precision** of context usage in LLM responses.  
  - 📌 **`ContextRecall.py`** – Evaluates the **recall** of context used by the LLM.  
  - 📌 **`ContextRecallFramework.py`** – Refactored version of **context precision & recall** tests.  
  - 📌 **`Faithfulness.py`** – Measures how **faithful** the LLM response is to the source content.  
  - 📌 **`RelevanceAndFactual.py`** – Checks **relevancy & factual correctness** of responses.  
  - 📌 **`TopicAdherence.py`** – Assesses how well the LLM **adheres to the given topic**.  
  - 📌 **`testDataFactory.py`** – Generates **test data** and pushes it to RAGAS.  
  - 📌 **`utils.py`** – Utility functions for **loading test data & fetching LLM responses**.  

---

## 🔑 Environment Variables

Create a `.env` file and add the following environment variables:

```ini
OPENAI_API_KEY=your_openai_api_key
RAGAS_APP_TOKEN=your_ragas_app_token
