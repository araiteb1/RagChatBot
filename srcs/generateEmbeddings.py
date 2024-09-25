from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


loaders = [PyPDFLoader('')]
docs = []

for file in loaders:
    docs.extend(file.load())


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)


embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})


vectorstore = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)


def batch_documents(documents, batch_size=166):
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]


for batch in batch_documents(docs, batch_size=166):
    vectorstore.add_documents(batch)


vectorstore.persist()


print(vectorstore._collection.count())


