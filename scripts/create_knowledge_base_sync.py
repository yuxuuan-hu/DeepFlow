import os
import time
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- Configuration ---
os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
CUSTOM_API_BASE = "YOUR_CUSTOM_API_BASE"

TUTORIALS_DIR = "your-tutorials-directory-path"
DATABASE_DIR = "your-database-directory-path"

TUTORIALS_INDEX_PATH = "faiss_index_tutorials"
DB_INDEX_PATH = "faiss_index_database"

def main():
    """Build static knowledge base index (synchronous version)."""
    print("--- Starting to build static knowledge base index (sync version)... ---")
    
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_base=CUSTOM_API_BASE, openai_api_key=os.environ['OPENAI_API_KEY'])
    except Exception as e:
        print(f"Error: Cannot initialize OpenAI Embeddings. Error details: {e}")
        return
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def process_source(source_path: str, index_path: str, glob_pattern: str = "**/*"):
        """Process data source and create FAISS index (synchronous)."""
        start_time = time.time()
        print(f"\n--- Processing source directory: {source_path} ---")
        
        if os.path.exists(index_path):
            print(f"Warning: Index '{index_path}' already exists. Please manually delete the folder if you want to rebuild.")
            # return

        if not os.path.isdir(source_path):
            print(f"Error: Directory '{source_path}' does not exist, skipping.")
            return

        print("Step 1/4: Loading files (this may take a few minutes)...")
        loader = DirectoryLoader(
            source_path, glob=glob_pattern, loader_cls=TextLoader, 
            use_multithreading=True, show_progress=True, silent_errors=True
        )
        docs = loader.load()
        
        if not docs:
            print("Error: No documents loaded.")
            return
        print(f"Loaded {len(docs)} document fragments.")

        print("Step 2/4: Splitting documents...")
        splits = text_splitter.split_documents(docs)
        print(f"Documents split into {len(splits)} chunks.")

        print("Step 3/4: Creating vector index (this may be very time-consuming)...")
        try:
            vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)
        except Exception as e:
            print(f"Error: Failed to create vector index. May be an API call issue. Error details: {e}")
            return
            
        print("Step 4/4: Saving index to disk...")
        vector_store.save_local(index_path)
        
        end_time = time.time()
        print(f"--- Successfully created and saved index to '{index_path}'! Total time: {end_time - start_time:.2f} seconds ---")

    # --- Execute processing ---
    process_source(TUTORIALS_DIR, TUTORIALS_INDEX_PATH)
    process_source(DATABASE_DIR, DB_INDEX_PATH, "**/*")  # Use "**/*" to match all files and subdirectories

    print("\n--- All static knowledge base processing completed! ---")


if __name__ == "__main__":
    main()
