# Description of the program:
# This program is designed to facilitate language translation using the LangChain and Groq APIs.
# It first sets up a FastAPI server and defines a system template for translation.
# A prompt template is created to process translation requests, and a ChatGroq model is initialized.
# The output of the translation is then parsed and displayed.
# The program adds a route to the FastAPI server for processing translation requests and runs the server.

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes


# 1.Create prompt template
system_template = "translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# 2.Create Model
model = ChatGroq(model="llama3-8b-8192")

# 3.Create parser
parser = StrOutputParser()

# 4.Create chain
chain = prompt_template | model | parser

# 4.App Definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChian's Runnable interfaces",
)

# 5.adding chain route
add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
