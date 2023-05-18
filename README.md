TO DO:
- lowkey, I think I'm going to have to take 5 steps back and change the model to an open-source one that I can run locally. 

- You can put a call back manager in the ChatOpenAI function
- Jsonify output as well!
    - output parsers
    - https://twitter.com/pwang_szn/status/1658774414681575426/photo/1
- Router Chains Select Index 
    - router is quietly the most important piece of the intelligence stack
    - top 40 then re-rank 
- Document loader setup to replace the load conversation function
- Multi-file indexing 
    - ETL (extract transform load)
        - Launch a container for each customer, not filter on customer 
- Will likely need some form of chaching 
- Would love to make it such that the system goals is always the beginning of every context window, and only once. But that is sort of pie in the sky shit rn. 

# Mem-Med

2. Create a chatbot Med-01, that asks about illness then provides a diagnosis

3. Integate the memory functionality into the Med-01 
- ChatModel Agent
    - https://twitter.com/LangChainAI/status/1657430519607623680/photo/2
- Integrate a ReAct structured Agent 

4. Create another chatbot (med-02), that indexes the memory of the other Med-01 and summarizes the input symptoms, the diagnosis, and potential alternative diagnosis. 

5. Convert into a streamlit app. 

6. Covert Memory records into the index 
- LLM Reranking in LlamaIndex
    - https://twitter.com/jerryjliu0/status/1657827898517012486
- Filter by Patient, can't mix patient info 

# Interesting Things to have around
Github Repo reader
- https://gpt-index.readthedocs.io/en/latest/examples/data_connectors/GithubRepositoryReaderDemo.html
Knowledge Graph
- https://gpt-index.readthedocs.io/en/latest/examples/index_structs/knowledge_graph/KnowledgeGraphDemo.html
- FLARE for Augmented Recall 
    - https://github.com/jzbjyb/FLARE
- Metaphor Langchain Intregration
    - https://twitter.com/LangChainAI/status/1657860892837683201/photo/1
- Load in for each person - ChatMessageHistory()
    - https://twitter.com/LangChainAI/status/1657864015480242176

Completed Tasks: 

1. Create Memory that consistently updates


