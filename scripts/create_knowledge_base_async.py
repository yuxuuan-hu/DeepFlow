import os
import time
import asyncio 
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

# --- Configuration ---
os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
CUSTOM_API_BASE = "YOUR_CUSTOM_API_BASE"

TUTORIALS_DIR = "your-tutorials-directory-path"
DATABASE_DIR = "your-database-directory-path"

TUTORIALS_INDEX_PATH = "faiss_index_tutorials"
DB_INDEX_PATH = "faiss_index_database"


async def process_source(source_path: str, index_path: str, glob_pattern: str = "**/*"):
    """
    Processes a single data source and creates a FAISS index asynchronously.
    """
    start_time = time.time()
    print(f"\n--- Processing source directory: {source_path} ---")

    print("Step 1/4: Loading files (this may take a few minutes)...")
    loader = DirectoryLoader(
        source_path, glob=glob_pattern, loader_cls=TextLoader, 
        use_multithreading=True, show_progress=True, silent_errors=True
    )
    docs = loader.load()
    if not docs:
        print("Error: No documents were loaded.")
        return
    print(f"Loaded {len(docs)} document fragments.")

    print("Step 2/4: Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Documents were split into {len(splits)} chunks.")

    print("Step 3/4: Creating vector index asynchronously (this can be very time-consuming)...")
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            openai_api_base=CUSTOM_API_BASE,
            openai_api_key=os.environ['OPENAI_API_KEY'],
            chunk_size=200
        )

        # Initialize index with the first document
        print("Initializing index with the first batch of documents...")
        vector_store = await FAISS.afrom_documents(documents=splits[:1], embedding=embeddings)

        # Add remaining documents in batches to avoid token limits
        print("Adding remaining documents in batches...")
        batch_size = 100
        remaining_splits = splits[1:]
        
        for i in tqdm(range(0, len(remaining_splits), batch_size), desc="Processing batches"):
            batch = remaining_splits[i:i+batch_size]
            try:
                await vector_store.aadd_documents(documents=batch)
            except Exception as batch_error:
                print(f"\nWarning: Error processing batch {i//batch_size + 1}: {batch_error}")
                print("Continuing with next batch...")
                continue

    except Exception as e:
        print(f"Error: Failed to create vector index. Details: {e}")
        return
        
    print("Step 4/4: Saving index to disk...")
    vector_store.save_local(index_path)
    
    end_time = time.time()
    print(f"--- Successfully created and saved index to '{index_path}'! Total time: {end_time - start_time:.2f} seconds ---")


async def main():
    """Main function to build the knowledge base indexes."""
    print("--- Starting to build static knowledge base index... ---")
    
    await process_source(TUTORIALS_DIR, TUTORIALS_INDEX_PATH)
    await process_source(DATABASE_DIR, DB_INDEX_PATH, "**/*")

    print("\n--- All static knowledge bases have been processed! ---")


if __name__ == "__main__":
    asyncio.run(main())
