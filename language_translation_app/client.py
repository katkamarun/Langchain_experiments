# Description of the program:
# This program is designed to facilitate language translation using the LangChain and Groq APIs.
# It first loads environment variables, including API keys for LangChain and Groq, and sets up the LangChain project and tracing.
# The program then initializes a ChatGroq model and defines initial messages for processing.
# It processes these initial messages and chains together the model and parser for further processing.
# A system template is defined for translation, and a prompt template is created to process translation requests.
# The output of the translation is then parsed and displayed.

from langserve import RemoteRunnable

# Initialize a remote chain for language translation
remote_chain = RemoteRunnable("http://localhost:8000/chain/")

# Invoke the remote chain with parameters for translation
remote_chain.invoke({"language": "italian", "text": "hi"})
