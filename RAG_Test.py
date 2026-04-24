from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from rich import print
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_mistralai  import MistralAIEmbeddings
from langchain.messages import HumanMessage

# loader = WebBaseLoader("https://www.apple.com/in/shop/buy-mac/macbook-air/13-inch")
# docs = loader.load()

llm = ChatMistralAI(
    model="mistral-small-latest", 
    temperature=0.7,
)

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriver = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
)

prompt = ChatPromptTemplate.from_messages(
    [
    ("system", "You are a helpful assistant. Use the context provided to answer the question. If you cannot find the answer in the context, say you don't know."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "context: {context}, Question: {question}"),
    ]
)

messages = []

print("Type 0 to exit")
while True:
    query = input("Enter your question: ")
    if query == "0":
        break

    ans = retriver.invoke(query)
    messages.append(HumanMessage(query))

    context = "\n\n".join([docs.page_content for docs in ans])

    final_prompt = prompt.invoke({"context": context, "question": query, "history": messages})

    finalAns = llm.invoke(final_prompt)
    messages.append(finalAns)

    print(finalAns.content)