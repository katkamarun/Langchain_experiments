import os
from dotenv import find_dotenv, load_dotenv
import getpass
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
if not LANGCHAIN_API_KEY:
    LANGCHAIN_API_KEY = getpass.getpass("Enter your LangChain API key: ")
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

LANGCHAIN_PROJECT = "PROJECT-01"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    GROQ_API_KEY = getpass.getpass("Enter your Groq API key: ")
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

model = ChatGroq(model="llama3-8b-8192")
messages = [
    SystemMessage(content="Translate the following from english into Spanish"),
    HumanMessage(content="hi!"),
]
response = model.invoke(messages)
# parser
parser = StrOutputParser()
parser.invoke(response)

# chaining
chain = model | parser
chain.invoke(messages)

system_template = "translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
result = prompt_template.invoke({"language": "italian", "text": "hi"})
result.to_messages()

# chaining together components with LCEL
chain = prompt_template | model | parser
chain.invoke({"language": "italian", "text": "how are you?"})
