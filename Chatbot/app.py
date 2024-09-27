"""
This program is a chatbot application that uses the LangChain API for natural language processing.
It is designed to assist users in a conversational manner, providing helpful responses to their queries.
The program is configured to remember the context of the conversation, ensuring a more coherent and personalized interaction.
"""

import os
from dotenv import find_dotenv, load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

from langchain_core.runnables.history import RunnableWithMessageHistory


# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Retrieve or prompt for LangChain API key
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Set LangChain project and tracing
LANGCHAIN_PROJECT = "PROJECT-02"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Retrieve or prompt for Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# model
model = ChatGroq(model="llama3-8b-8192")
model.invoke([HumanMessage(content="hi! I'm Bob")])
model.invoke(
    [HumanMessage(content="what's my name?")]
)  # model wont remember the context

# we need to make sure it remembers context
model.invoke(
    [
        HumanMessage(content="hi! I'm Bob"),
        AIMessage(content="hello Bob! How can I assist you today?"),
        HumanMessage(content="what's my name?"),
    ]
)

# message History
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieves or initializes a session history for a given session ID.

    Args:
        session_id (str): The unique identifier for the session.

    Returns:
        BaseChatMessageHistory: The session history object associated with the session ID.
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(model, get_session_history)

config = {"configurable": {"session_id": "abc2"}}

response = with_message_history.invoke(
    [HumanMessage(content="hi! I'm Bob")],
    config=config,
)

response.content

response = with_message_history.invoke(
    [HumanMessage(content="What's my name")],
    config=config,
)

response.content


config = {"configurable": {"session_id": "abc3"}}

response = with_message_history.invoke(
    [HumanMessage(content="What's my name")],
    config=config,
)

response.content

config = {"configurable": {"session_id": "abc2"}}

response = with_message_history.invoke(
    [HumanMessage(content="What's my name")],
    config=config,
)

response.content


# prompt templates
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

response = chain.invoke({"messages": [HumanMessage(content="hi! I'm Bob")]})
response.content

with_message_history = RunnableWithMessageHistory(chain, get_session_history)
config = {"configurable": {"session_id": "abc5"}}

response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Jim")],
    config=config,
)

response.content

response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)

response.content


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model


response = chain.invoke(
    {"messages": [HumanMessage(content="hi!, I 'm Bob")], "language": "Spanish"}
)

response.content


with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc11"}}

response = with_message_history.invoke(
    {"messages": [HumanMessage(content="hi! I'm todd")], "language": "Spanish"},
    config=config,
)

response.content


response = with_message_history.invoke(
    {"messages": [HumanMessage(content="whats my name?")], "language": "Spanish"},
    config=config,
)

response.content

# managing conversation history

from langchain_core.messages import SystemMessage, trim_messages

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    # HumanMessage(content="hi! I'm bob"),
    # AIMessage(content="hi!"),
    # HumanMessage(content="I like vanilla ice cream"),
    # AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

trimmer.invoke(messages)


from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my name?")],
        "language": "English",
    }
)
response.content

response = chain.invoke(
    {
        "messages": messages
        + [HumanMessage(content="what is the math problem did i ask?")],
        "language": "English",
    }
)
response.content

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc20"}}


response = with_message_history.invoke(
    {
        "messages": messages + [HumanMessage(content="whats my name?")],
        "language": "English",
    },
    config=config,
)

response.content

response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="what math problem did i ask?")],
        "language": "English",
    },
    config=config,
)

response.content


# streaming
config = {"configurable": {"session_id": "abc15"}}
for r in with_message_history.stream(
    {
        "messages": [HumanMessage(content="hi! I'm todd. tell me a joke")],
        "language": "English",
    },
    config=config,
):
    print(r.content, end="|")
