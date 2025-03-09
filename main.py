import os
import streamlit as st
import pickle
import time
import faiss
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
st.title("News Research Tool")
st.sidebar.title("News Article URLs")
urls= []
file_path = "vector_index.pkl"
llm=OpenAI(temperature=0.9, max_tokens=500)

#create embeddings
# Create the embeddings of the chunks using openAIEmbeddings
embeddings = OpenAIEmbeddings() 

for i in range(2):
    url= st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)


click_button=st.sidebar.button("Process URLs")
main_placeholder=st.empty()
if click_button:
    loader= UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data loading....Started.....")
    data= loader.load()
    text_splitter=RecursiveCharacterTextSplitter(
                        separators=['\n\n','\n','.',','],
                        chunk_size=500,
                        chunk_overlap=50
                    )
    main_placeholder.text("Data splitting....Started.....")
    docs=text_splitter.split_documents(data)

    # Pass the documents and embeddings inorder to create FAISS vector index
    vectorindex_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Data embedding....Started.....")
    time.sleep(2)
    # Save the FAISS index separately
    faiss.write_index(vectorindex_openai.index, "faiss_index.bin")

    # Save embeddings and document metadata separately using pickle
    with open(file_path, "wb") as f:
        pickle.dump((vectorindex_openai.index_to_docstore_id, vectorindex_openai.docstore, vectorindex_openai.index), f)
        
query=main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorIndex = pickle.load(f)  
        
        with open(file_path, "rb") as f:
            index_to_docstore_id, docstore, faiss_index = pickle.load(f)

        # Ensure you pass the required embedding function
        vectorIndex = FAISS(
            index=faiss_index, 
            docstore=docstore, 
            index_to_docstore_id=index_to_docstore_id, 
            embedding_function=embeddings  # Ensure you have this defined
        )

        print("FAISS index loaded successfully!")      
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
        result= chain.invoke({"question": query}, return_only_outputs=True)
        #result will be dictionary->{answer:"",sources:""}
        st.header("Answer")
        st.text(result["answer"])
        
        sources=result.get("sources","")
        if sources:
            st.subheader("Sources:")
            sources_list=sources.split("\n")
            for source in sources_list:
                st.write(source)
                