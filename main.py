import os
import gradio as gr
from dotenv import load_dotenv


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader

load_dotenv()


class VectorStore:
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Initialize/Connect to the Vector DB
        # This will load existing data if './chroma_db' exists
        self.db = Chroma(
            collection_name="demo_collection",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db",
        )
        print(
            f"✅ Vector Store Initialized. Current doc count: {self.db._collection.count()}"
        )

    def add_documents(self, file_paths):
        """Loads files, splits them, and adds them to the existing vector DB."""
        documents = []

        # 1. Load the actual content from files
        for file_path in file_paths:
            try:
                if file_path.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif file_path.endswith(".txt"):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if not documents:
            return "No valid text found in files."

        # 2. Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        # 3. Add to Vector Store (Appending, not overwriting)
        self.db.add_documents(chunks)

        return f"Added {len(chunks)} chunks to the database."

    def as_retriever(self):
        # Return a retriever object for the chain
        return self.db.as_retriever(search_kwargs={"k": 3})


rag_store = VectorStore()


def process_uploaded_files(files):
    """
    Placeholder function to process uploaded files.
    In the next step, you will add logic here to:
    1. Read the file content
    2. Split text into chunks
    3. Create embeddings and store them in a vector database
    """
    if not files:
        return "No files uploaded.", gr.update(interactive=True)

    file_paths = [f.name for f in files]
    file_names = [os.path.basename(f) for f in file_paths]

    # Add to our global store
    status_msg = rag_store.add_documents(file_paths)

    return (
        f"Processed: {', '.join(file_names)}. \n{status_msg}",
        True,  # Set files_ready to True
    )


def chat_function(message, history, files_processed):
    """
    Placeholder function for the chat logic.
    In the next step, you will add logic here to:
    1. Search the vector database for relevant chunks based on 'message'
    2. Send the chunks + message to an LLM
    3. Return the answer
    """
    if not files_processed:
        return "Please upload and process documents first."

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    retriever = rag_store.as_retriever()

    # 3. Create RAG Chain
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(message)
    return response


# --- Gradio UI Layout ---
with gr.Blocks(title="RAG Chatbot") as app:
    gr.Markdown("# 📄 RAG Chatbot: Chat with your Data")

    # State variable to track if files are ready
    files_ready = gr.State(False)

    with gr.Row():
        # Left Column: Upload Area
        with gr.Column(scale=1):
            gr.Markdown("### Step 1: Upload Documents")
            file_input = gr.File(
                file_count="multiple",
                label="Upload PDFs or Text Files",
                file_types=[".pdf", ".txt"],
            )
            process_btn = gr.Button("Process Documents", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)

        # Right Column: Chat Area
        with gr.Column(scale=2):
            gr.Markdown("### Step 2: Chat")
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(
                placeholder="Ask a question about your documents...", interactive=True
            )
            clear = gr.ClearButton([msg, chatbot])

    # --- Interaction Logic ---

    # 1. Processing files
    process_btn.click(
        fn=process_uploaded_files,
        inputs=[file_input],
        outputs=[status_output, files_ready],
    )

    # 2. Chatting
    def respond(message, chat_history, files_status):
        # Determine the bot's response
        bot_message = chat_function(message, chat_history, files_status)

        # Append user message and bot response to history
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})

        return "", chat_history

    msg.submit(fn=respond, inputs=[msg, chatbot, files_ready], outputs=[msg, chatbot])

if __name__ == "__main__":
    app.launch()
