# Description of the program:
# This program is designed to facilitate language translation using the LangChain and Groq APIs.
# It first loads environment variables, including API keys for LangChain and Groq, and sets up the LangChain project and tracing.
# The program then initializes a ChatGroq model and defines initial messages for processing.
# It processes these initial messages and chains together the model and parser for further processing.
# A system template is defined for translation, and a prompt template is created to process translation requests.
# The output of the translation is then parsed and displayed.

import os
from dotenv import find_dotenv, load_dotenv
import getpass
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Retrieve or prompt for LangChain API key
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
if not LANGCHAIN_API_KEY:
    LANGCHAIN_API_KEY = getpass.getpass("Enter your LangChain API key: ")
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

# Set LangChain project and tracing
LANGCHAIN_PROJECT = "PROJECT-01"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Retrieve or prompt for Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    GROQ_API_KEY = getpass.getpass("Enter your Groq API key: ")
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize ChatGroq model
model = ChatGroq(model="llama3-8b-8192")

# Define initial messages
messages = [
    SystemMessage(content="Translate the following from english into Spanish"),
    HumanMessage(content="hi!"),
]

# Process initial messages
response = model.invoke(messages)
parser = StrOutputParser()
parser.invoke(response)

# Chain model and parser for processing
chain = model | parser
chain.invoke(messages)

# Define a system template for translation
system_template = "translate the following into {language}:"

# Create a prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# Process a translation request
result = prompt_template.invoke({"language": "italian", "text": "hi"})
result.to_messages()

# Chain together components for LCEL processing
chain = prompt_template | model | parser
chain.invoke({"language": "italian", "text": "how are you?"})
