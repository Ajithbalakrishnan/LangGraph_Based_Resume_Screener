# LangGraph Based Resume Screener
An LLM Chatbot based on LangGraph and LangChain that dynamically retrieves and processes resumes using RAG to perform resume screening.

# Introduction to the Project
**Project Overview**
This project endeavors to showcase a Proof of Concept (POC) of a Language Model (LLM) chatbot designed to revolutionize the resume screening process for hiring managers. Traditional methods rely heavily on keyword-based screening, often overlooking nuanced candidate qualifications and potential. In contrast, the proposed chatbot harnesses advanced LLM capabilities to handle unstructured, natural language data found in job descriptions and resumes, performing tasks akin to human recruiters but with greater efficiency and accuracy.

**Need for Innovation**
Despite the surge in job applications annually, current screening tools remain limited in their ability to effectively process and evaluate candidates' qualifications. Keyword-centric approaches struggle with the intricacies of human-written documents, leading to suboptimal candidate matches and prolonged hiring cycles. This underscores a critical need for integrating LLM-based technologies into the recruitment process to enhance precision and streamline operations.

# Role of RAG Frameworks
**Enhancing Chatbot Reliability**
RAG (Retriever-Augmented Generation) frameworks emerge as pivotal enhancements for chatbot reliability. By integrating external knowledge bases with LLM agents, RAG equips chatbots to contextualize user queries comprehensively. This augmentation significantly bolsters the accuracy and relevance of responses, particularly in data-intensive fields like recruitment where precise information retrieval is paramount.

**Addressing Complexity in Human Communication:**
LLM Agents with RAG capabilities excel in deciphering intricate and ambiguous human prompts commonly found in job descriptions and resumes. While LLM generators adeptly handle such complexities, retrievers may struggle with multifaceted queries, impacting the quality of resume matches. Leveraging RAG frameworks can mitigate this challenge by improving the precision of resume retrieval, thereby optimizing candidate screening outcomes.


**Solution Over view:**
![Screenshot_125](https://github.com/Ajithbalakrishnan/LangGraph_Based_Resume_Screener/blob/main/assets/Solution_Overview.jpg)

**User Interface:**
![Screenshot_125](https://github.com/Ajithbalakrishnan/LangGraph_Based_Resume_Screener/blob/main/assets/UI.jpg)


**Based on:** 
- `streamlit`: Streamlit turns data scripts into shareable web apps in minutes.
All in pure Python. No front‑end experience required.
- `FaISS` : FAISS (Facebook AI Similarity Search) is an open-source library that helps developers quickly search for similar multimedia documents, such as images, videos, and text, within large datasets. It can also be used to cluster dense vectors. FAISS can be used to build indexes and perform searches with high speed and memory efficiency. 
- `Langchain` : LangChain is a framework designed to simplify the creation of applications using large language models. As a language model integration framework, LangChain's use-cases largely overlap with those of language models in general, including document analysis and summarization, chatbots, and code analysis.
- `hugginface`: For Embedding models   

- `Openai` `Gemnai` `HF Space` `Ollama`- LLM api integration
- ` LangGraph` - LLM agent integration (Yet to be implemented)


## How to Use Me?
Install MiniConda:
```
https://docs.anaconda.com/miniconda/miniconda-install/
```
Setup Conda environment
```
conda env create -f environment.yml
conda activate <env name>
```
Run the solution 
```
streamlit run src\main.py
```

