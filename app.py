#from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st 
import os
from dotenv import load_dotenv 

load_dotenv()

sec_key=os.getenv("hugging_face_token")

print(sec_key)

import os 
os.environ["HUGGINGFACEHUB_API_TOKEN"]=sec_key

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["Langchain_token_key"]=os.getenv("Langchain_token_key")

repo_id="Mistralai/Mistral-7B-Instruct-v0.3"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=sec_key)

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user", "Question:{question}")
    ]
      
    )

st.title("Langchain Demo With MISTRALAI API")
input_test=st.text_input("Search the topic you want")

#llm=ChatOpenAI(model="gpt-3.5-turbo")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_test:
    st.write(chain.invoke({'question':input_test}))