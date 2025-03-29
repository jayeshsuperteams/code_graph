from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

def rag_query(question: str, top_k=3):
    # Load documents
    loader = TextLoader("example.txt")
    documents = loader.load()
    
    # Split text
    text_splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        separator="\n\n",
        length_function=lambda x: len(x.split())
    )
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'truncation': True, 'max_length': 512}
    )
    
    # Create vector store
    vector_store = FAISS.from_documents(texts, embeddings)
    
    # Generate response
    docs = vector_store.similarity_search(question, k=top_k)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""Context: {context}
    Question: {question}
    Answer:"""
    
    generator = pipeline('text-generation', model='gpt2-xl', max_new_tokens=150)
    return generator(prompt)[0]['generated_text']
