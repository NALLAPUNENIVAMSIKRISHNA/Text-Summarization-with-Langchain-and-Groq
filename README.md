# Text Summarization using LangChain and ChatGroq

## Description
This project performs text summarization using the **LangChain** framework and the **ChatGroq** API with the **Llama3-8b-8192** model. The implementation explores different techniques for text summarization, leveraging open-source language models instead of OpenAI models.

## Features
- Uses **LangChain** for structured LLM interactions.
- Implements summarization using **ChatGroq API**.
- Utilizes **Llama3-8b-8192**, an open-source language model.
- Demonstrates different summarization techniques including **Split**, **Map-Reduce**, and **Refine** chains.
- Uses various prompt engineering techniques for optimized summarization.

## Installation
### Prerequisites
- Python 3.8+
- An active **Groq API Key** (stored in `.env` file)
- Required Python packages

### Steps
1. Clone the repository:
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file in the project root.
   - Add your **Groq API Key**:
     ```sh
     GROQ_API_KEY=your_api_key_here
     ```
4. Run the Jupyter Notebook:
   ```sh
   jupyter notebook text_summarization.ipynb
   ```

## Working of the Code
### 1. Importing Required Libraries
The notebook starts by importing necessary libraries:
```python
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
```
- `dotenv` is used to load environment variables from a `.env` file.
- `ChatGroq` from `langchain_groq` is used to interact with the **Groq API**.
- `load_summarize_chain` is used to implement different summarization chains.

### 2. Loading API Key
```python
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
```
- The `.env` file stores the API key securely.
- `os.getenv("GROQ_API_KEY")` retrieves the API key for authentication.

### 3. Initializing the Language Model
```python
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
```
- This initializes the **Llama3-8b-8192** model using the Groq API.
- `streaming=True` allows for real-time text generation.

### 4. Implementing Summarization Using Different Chains
#### **1. Split Chain Summarization**
- The input text is divided into chunks before summarization.
```python
split_chain = load_summarize_chain(llm, chain_type="stuff")
summary = split_chain.run(text)
```
- Best for small texts that can be processed at once.

#### **2. Map-Reduce Chain Summarization**
- Each chunk is summarized separately and combined in a second step.
```python
map_reduce_chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = map_reduce_chain.run(text)
```
- Suitable for long documents.

#### **3. Refine Chain Summarization**
- The model iteratively refines the summary as more chunks are processed.
```python
refine_chain = load_summarize_chain(llm, chain_type="refine")
summary = refine_chain.run(text)
```
- Best for generating a more detailed summary.

### 5. Exploring Different Prompt Templates
Various prompt templates were used for improved summarization

## Dependencies
The required dependencies can be installed using:
```sh
pip install langchain_groq python-dotenv jupyter
```

## References
- LangChain Summarization Guide: [LangChain Docs](https://python.langchain.com/v0.1/docs/use_cases/summarization/)
- LangChain Summarization Tutorial: [LangChain Tutorial](https://python.langchain.com/docs/tutorials/summarization/)

