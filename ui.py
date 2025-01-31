# import streamlit as st
# import os
# from langchain.document_loaders import PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain_mistralai import ChatMistralAI
# from langchain.prompts import PromptTemplate

# # Load PDF documents (this assumes your 'pdf rag' folder is mounted and accessible)
# def load_docs(directory):
#     loader = PyPDFDirectoryLoader(directory)
#     documents = loader.load()
#     return documents

# # Split the documents into chunks
# def split_docs(documents, chunk_size=512, chunk_overlap=20):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\\n", " ", ""]
#     )
#     docs = text_splitter.split_documents(documents)
#     return docs

# # Function to get similar documents based on query
# def get_similar_docs(query, k=4):
#     similar_docs = index.similarity_search(query, k=k)
#     return similar_docs

# # Function to get the answer using LangChain and Mistral model
# def get_answer(query):
#     relevant_docs = get_similar_docs(query)
#     response = chain.invoke({"input_documents": relevant_docs, "question": query})
#     return response['output_text']

# # Load and prepare data
# directory = 'pdf rag/'  # Make sure this path is correct
# documents = load_docs(directory)
# docs = split_docs(documents)

# # Create embeddings and FAISS index
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# index = FAISS.from_documents(docs, embeddings)

# # Load the Mistral AI model
# llm = ChatMistralAI(model='mistral-large-latest', temperature=0)

# # Define the prompt for question-answering
# med_prompt = PromptTemplate.from_template(
#     '''
#     You are an expert mathematics assistant. Based strictly on the provided source material, answer the following question clearly and with detailed steps.

#     Break down the solution step by step.
#     Explain the reasoning behind each step.
#     If the answer provided by the student is wrong then explain what mistake the student must have made and the misconception the student must be having.
#     For every incorrect option provided explain in steps what misconception the student must be having.
#     Only use the source material for your answer; do not generate any information outside of it.
#     If the information required to answer the question is not available in the source material, respond with "I don't know."

#     Source material: {context}

#     Question: {question}
#     Answer:
#     '''
# )

# # Create the LangChain question-answering chain
# chain = load_qa_chain(llm, chain_type="stuff", prompt=med_prompt)

# # Streamlit UI
# st.title("Mathematics Question Assistant")

# # Input: Question
# question = st.text_area("Enter the question")

# # Input: Options
# option_a = st.text_input("Option A")
# option_b = st.text_input("Option B")
# option_c = st.text_input("Option C")
# option_d = st.text_input("Option D")

# # Input: Student's Answer
# student_answer = st.selectbox("Student's Answer", ("A", "B", "C", "D"))

# # Button to process the question
# if st.button("Get Answer"):
#     # Formulate the question with options and student's answer
#     full_query = f"{question} Options a. {option_a} Options b. {option_b} Options c. {option_c} Options d. {option_d} Answer marked by student is Option {student_answer}."
    
#     # Get the response
#     answer = get_answer(full_query)
    
#     # Display the answer
#     st.write("### Explanation:")
#     st.write(answer)













import streamlit as st
import os
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate

# Load PDF documents (this assumes your 'pdf rag' folder is mounted and accessible)
def load_docs(directory):
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    return documents

# Split the documents into chunks
def split_docs(documents, chunk_size=512, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents)
    return docs

# Function to get similar documents based on query
def get_similar_docs(query, k=4):
    similar_docs = index.similarity_search(query, k=k)
    return similar_docs

# Function to get the answer using LangChain and Mistral model
def get_answer(query):
    relevant_docs = get_similar_docs(query)
    response = chain.invoke({"input_documents": relevant_docs, "question": query})
    return response['output_text']

# Load and prepare data
directory = 'pdf rag/'  # Make sure this path is correct
documents = load_docs(directory)
docs = split_docs(documents)

# Create embeddings and FAISS index
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
index = FAISS.from_documents(docs, embeddings)

# Load the Mistral AI model using the API key from environment variable
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    st.error("MISTRAL_API_KEY environment variable not set.")
else:
    llm = ChatMistralAI(api_key=mistral_api_key, model='mistral-large-latest', temperature=0)

    # Define the prompt for question-answering
    med_prompt = PromptTemplate.from_template(
        '''
        You are an expert mathematics assistant. Based strictly on the provided source material, answer the following question clearly and with detailed steps.

        Break down the solution step by step.
        Explain the reasoning behind each step.
        If the answer provided by the student is wrong then explain what mistake the student must have made and the misconception the student must be having.
        For every incorrect option provided explain in steps what misconception the student must be having.
        Only use the source material for your answer; do not generate any information outside of it.
        If the information required to answer the question is not available in the source material, respond with "I don't know."

        Source material: {context}

        Question: {question}
        Answer:
        '''
    )

    # Create the LangChain question-answering chain
    chain = load_qa_chain(llm, chain_type="stuff", prompt=med_prompt)

    # Streamlit UI
    st.title("Mathematics Question Assistant")

    # Input: Question
    question = st.text_area("Enter the question")

    # Input: Options
    option_a = st.text_input("Option A")
    option_b = st.text_input("Option B")
    option_c = st.text_input("Option C")
    option_d = st.text_input("Option D")

    # Input: Student's Answer
    student_answer = st.selectbox("Student's Answer", ("A", "B", "C", "D"))

    # Button to process the question
    if st.button("Get Answer"):
        # Formulate the question with options and student's answer
        full_query = f"{question} Options a. {option_a} Options b. {option_b} Options c. {option_c} Options d. {option_d} Answer marked by student is Option {student_answer}."
        
        # Get the response
        answer = get_answer(full_query)
        
        # Display the answer
        st.write("### Explanation:")
        st.write(answer)
