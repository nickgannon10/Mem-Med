from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os 

load_dotenv()

docs_dir = "docs"

embedding = OpenAIEmbeddings()

retrievers_info = []

# retrievers = []
# retriever_description = []
# retriever_names = []

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap  = 20,
    length_function = len,
)

for filename in os.listdir(docs_dir):
    doc = None
    if os.path.exists(filename[:-4]):
        retriever = Chroma(persist_directory=filename[:-4], embedding_function=embedding).as_retriever()
    else:
        if filename.endswith(".txt"):
            try: 
                with open(os.path.join(docs_dir, filename), "r", encoding="utf-8") as f:
                    doc = f.read()
            except UnicodeDecodeError:
                print(f"Could not read {filename}")
                continue
            doc = text_splitter.create_documents([doc])
        elif filename.endswith(".PDF") or filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(docs_dir, filename))
            doc = loader.load_and_split()
            print(doc)

        if doc is not None: 
            retriever = Chroma.from_documents(documents=doc, embedding=embedding, persist_directory=filename[:-4])
            retriever.persist()

    retriever_info = {
        'name': filename[:-4],
        'description': f"Good for answering questions about {filename}",
        'retriever': retriever
    }
    retrievers_info.append(retriever_info)

    # retrievers.append(retriever)
    # retriever_names.append(filename[:-4])
    # retriever_description.append(f"Good for answering questions about {filename}")

chain = MultiRetrievalQAChain.from_retrievers(OpenAI(), retrievers_info, verbose=True)

while True:
    print(chain.run(input("Ask a question: ")))




