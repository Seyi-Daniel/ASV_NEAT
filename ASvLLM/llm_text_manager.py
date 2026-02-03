import os
import re
from typing import Optional, List

# LangChain Imports
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- CONFIGURATION ---
PDF_DIRECTORY = "pdf_files/"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
VECTOR_STORE_PATH = "faiss_multi_pdf_index"

# --- API Key Environment Variables ---
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
ANTHROPIC_API_KEY_ENV = "CLAUDE_API_KEY"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"


class LLMTextManager:
    """
    Manages the RAG (text-only) pipeline.
    It can be initialized to use OpenAI, Claude, or DeepSeek.
    """

    def __init__(self, provider: str = "openai"):
        print(f"Initializing LLMTextManager for provider: {provider}...")
        self.provider = provider
        self.llm = self._setup_llm(provider)
        if self.llm is None:
            raise ValueError(f"Failed to initialize LLM. Check API key for {provider}.")

        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = self._load_or_build_vector_store()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        self.rag_chain = self._setup_rag_chain()
        print("LLMTextManager is ready.")



    def _setup_llm(self, provider: str):
        """Initializes the correct LangChain LLM object based on the provider."""
        if provider == "openai":
            api_key = os.getenv(OPENAI_API_KEY_ENV)
            if not api_key: raise ValueError(f"{OPENAI_API_KEY_ENV} not set.")
            return ChatOpenAI(model="gpt-4o", openai_api_key=api_key, max_tokens=1000, temperature=0.1)

        elif provider == "claude":
            api_key = os.getenv(ANTHROPIC_API_KEY_ENV)
            if not api_key: raise ValueError(f"{ANTHROPIC_API_KEY_ENV} not set.")
            return ChatAnthropic(model="claude-3-opus-20240229", anthropic_api_key=api_key, max_tokens=1000,
                                 temperature=0.1)

        elif provider == "deepseek":
            api_key = os.getenv(DEEPSEEK_API_KEY_ENV)
            if not api_key: raise ValueError(f"{DEEPSEEK_API_KEY_ENV} not set.")
            return ChatOpenAI(
                model="deepseek-chat",
                openai_api_key=api_key,
                openai_api_base="https://api.deepseek.com/v1",
                max_tokens=1000,
                temperature=0.1
            )
        else:
            raise ValueError(f"Unknown LLM provider for RAG: {provider}")

    def _load_or_build_vector_store(self) -> FAISS:
        """Loads or builds the FAISS vector store."""
        if os.path.exists(VECTOR_STORE_PATH):
            print(f"Loading existing vector store from {VECTOR_STORE_PATH}...")
            try:
                return FAISS.load_local(VECTOR_STORE_PATH, self.embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"Warning: Failed to load vector store: {e}. Rebuilding...")

        print(f"No existing vector store found. Building from all PDFs in {PDF_DIRECTORY}...")
        all_docs: List[Document] = []
        if not os.path.isdir(PDF_DIRECTORY):
            raise FileNotFoundError(f"The specified PDF directory does not exist: {PDF_DIRECTORY}")
        pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith(".pdf")]
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in the directory: {PDF_DIRECTORY}")

        print(f"Found {len(pdf_files)} PDF files to process.")
        for pdf_file in pdf_files:
            pdf_path = os.path.join(PDF_DIRECTORY, pdf_file)
            print(f" > Loading documents from {pdf_path}...")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)

        if not all_docs:
            raise ValueError(f"Could not load any documents from the PDFs in {PDF_DIRECTORY}.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        split_docs = text_splitter.split_documents(all_docs)
        print(f"Creating embeddings for {len(split_docs)} document chunks...")
        vector_store = FAISS.from_documents(split_docs, self.embeddings)

        try:
            vector_store.save_local(VECTOR_STORE_PATH)
            print(f"Vector store saved to {VECTOR_STORE_PATH}")
        except Exception as e:
            print(f"Warning: Failed to save vector store: {e}")

        return vector_store

    def _setup_rag_chain(self):
        """Sets up the LangChain Expression Language (LCEL) chain."""
        template = """
        You are a ship navigation expert system. Your task is to determine the correct action for each vessel based on the International Regulations for Preventing Collisions at Sea (COLREGs).
        First, use the following relevant COLREGs rules to inform your decision:
        --- RELEVANT COLREGS RULES ---
        {context}
        ------------------------------
        Now, analyze the following real-time vessel data and determine the situation, role, and required action for each vessel. The prompt already contains instructions for the output format.
        --- VESSEL DATA AND INSTRUCTIONS ---
        {dynamic_prompt}
        ------------------------------------
        """
        prompt = ChatPromptTemplate.from_template(template)
        output_parser = StrOutputParser()

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
                RunnableParallel(
                    {"context": RunnablePassthrough() | self.retriever | format_docs,
                     "dynamic_prompt": RunnablePassthrough()}
                )
                | prompt
                | self.llm
                | output_parser
        )
        return chain

    def _clean_json_response(self, content: str) -> Optional[str]:
        """Extracts a JSON array from the LLM's messy output."""
        if not isinstance(content, str): return None
        match = re.search(r'```(json)?\s*(\[.*\])\s*```', content, re.DOTALL)
        if match: return match.group(2)
        start = content.find('[')
        end = content.rfind(']')
        if start != -1 and end != -1 and end > start: return content[start:end + 1]
        return None

    def get_llm_decision_standard(self, prompt: str) -> Optional[str]:
        """
        Gets a decision from the LLM *without* RAG enrichment.
        Just sends the prompt directly.
        """
        if not self.llm:
            print("ERROR: LLM not initialized.")
            return None

        print(f"\n--- Standard Manager ({self.provider}): Sending prompt directly... ---")
        try:
            # Call the LLM directly
            response = self.llm.invoke(prompt)
            # Extract text content (langchain response object)
            raw_response = response.content

            clean_json_string = self._clean_json_response(raw_response)
            if not clean_json_string:
                print(f"--- Standard Warning: Could not extract JSON. ---")
                return "[]"
            return clean_json_string
        except Exception as e:
            print(f"Error during Standard execution: {e}")
            return "[]"

    def get_llm_decision(self, dynamic_prompt: str) -> Optional[str]:
        """Gets a decision from the LLM based on the RAG-enriched prompt."""
        if not self.rag_chain:
            print("ERROR: RAG chain is not initialized.")
            return None
        print(f"\n--- RAG Manager ({self.provider}): Enriching prompt... ---")
        try:
            raw_response = self.rag_chain.invoke(dynamic_prompt)
            clean_json_string = self._clean_json_response(raw_response)
            if not clean_json_string:
                print(f"--- RAG Warning ({self.provider}): Could not extract JSON. ---")
                print("Raw response was:", raw_response)
                return "[]"
            return clean_json_string
        except Exception as e:
            print(f"Error during RAG chain execution ({self.provider}): {e}")
            return "[]"