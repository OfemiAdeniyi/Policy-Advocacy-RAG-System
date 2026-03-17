import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. SETUP: Your API Key
os.environ["GROQ_API_KEY"] = "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"

print("Loading the policy database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./policy_db_local", embedding_function=embeddings)

# 3. SETUP THE AI (Using Llama 3.1)
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

# 4. DESIGN THE PROMPT: Telling the AI how to behave
system_prompt = (
    "You are an expert on Nigerian Public Policy. "
    "Use the following retrieved context to answer the user's question. "
    "If the answer is not in the context, say that you don't know based on the documents provided. "
    "Keep your answers professional and cite the policy name if possible."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 5. CREATE THE CHAIN: Linking the DB, the Prompt, and the AI
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 5}), question_answer_chain)

# 6. INTERACTIVE LOOP: The User Interface
print("\n--- Policy.ng AI Assistant is Live ---")
print("Type 'quit' to exit.")

while True:
    user_query = input("\nAsk a policy question: ")
    
    if user_query.lower() == 'quit':
        break
    
    if not user_query.strip():
        continue

    print("Searching the archives...")
    response = rag_chain.invoke({"input": user_query})
    
    print("\nAI RESPONSE:")
    print(response["answer"])