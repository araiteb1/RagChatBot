import os
import signal
import sys

import google.generativeai as genai
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

GEMEINI_API_KEY = ""

def handlerSignal(sig, frame):
    print("\n Thanks you. :) ")
    sys.exit(0)

signal.signal(signal.SIGINT, handlerSignal)

def generateRagPrompt(query, context):
    escaped = context.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = ("""
        If the context is irrelevant to the answer, you may ignore it.
                QUESTION: '{query}'
                CONTEXT:'{context}'


                ANSWER: 
                """).format(query=query, context=context)

    return prompt

def get_DataInfirmation_From_db(query):
    context = ""
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)
    search_results = vector_db.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    return context

def generateAnswer(prompt):
    genai.configure(api_key=GEMEINI_API_KEY)
    model = genai.GenerativeModel(model_name='gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text


welcome_text = generateAnswer("Can you quickly introduce yourself")
print(welcome_text)

while True:
    print("--------------------------------------------------")
    print("What is your Question?")
    query = input("Query: ")
    context = get_DataInfirmation_From_db(query)
    prompt = generateRagPrompt(query=query, context=context)
    answer = generateAnswer(prompt=prompt)
    print(answer)
