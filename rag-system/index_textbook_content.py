"""
Script to index all textbook content into the RAG system
This will read all markdown files from the docs directory and index them
"""
import os
import sys
import glob
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the api directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

def read_markdown_files(docs_dir):
    """Read all markdown files from the docs directory"""
    print(f"🔍 Reading markdown files from: {docs_dir}")

    md_files = []
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                md_files.append(file_path)

    print(f"📁 Found {len(md_files)} markdown files")
    return md_files

def extract_content_from_file(file_path):
    """Extract content from a markdown file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Get the relative path for the chapter_id
        relative_path = os.path.relpath(file_path, os.path.dirname(os.path.dirname(__file__)))
        chapter_id = relative_path.replace(os.sep, '_').replace('.md', '')

        # Get just the filename for the chapter title
        chapter_title = os.path.basename(file_path).replace('.md', '').replace('_', ' ').title()

        return {
            'chapter_id': chapter_id,
            'chapter_title': chapter_title,
            'content': content,
            'file_path': file_path
        }
    except Exception as e:
        print(f"❌ Error reading {file_path}: {str(e)}")
        return None

def chunk_text(text, max_chunk_size=1000):
    """Split text into chunks of approximately max_chunk_size words"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_chunk_size):
        chunk = ' '.join(words[i:i + max_chunk_size])
        chunks.append(chunk)

    return chunks

def index_content_to_rag():
    """Index all textbook content to the RAG system"""
    try:
        from api.main import app, ingest_content, IngestRequest
        import asyncio

        # Get the docs directory
        docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')

        if not os.path.exists(docs_dir):
            print(f"❌ Docs directory not found: {docs_dir}")
            return False

        # Read all markdown files
        md_files = read_markdown_files(docs_dir)

        if not md_files:
            print("❌ No markdown files found to index")
            return False

        total_chunks = 0
        successful_ingests = 0

        print(f"\n📦 Starting content indexing...")

        for file_path in md_files:
            print(f"   Processing: {os.path.basename(file_path)}")

            file_data = extract_content_from_file(file_path)
            if not file_data:
                continue

            # Chunk the content
            content_chunks = chunk_text(file_data['content'])
            print(f"     Split into {len(content_chunks)} chunks")

            for i, chunk in enumerate(content_chunks):
                # Create a unique ID for this chunk
                chunk_id = f"{file_data['chapter_id']}_chunk_{i}"

                # Create ingest request
                ingest_req = IngestRequest(
                    content=chunk,
                    chapter_id=chunk_id,
                    chapter_title=f"{file_data['chapter_title']} - Part {i+1}",
                    metadata={
                        "original_file": file_data['file_path'],
                        "chunk_index": i,
                        "total_chunks": len(content_chunks),
                        "source": "textbook"
                    }
                )

                try:
                    # Ingest the content
                    result = asyncio.run(ingest_content(ingest_req, None))

                    if result.success:
                        successful_ingests += 1
                        total_chunks += 1
                    else:
                        print(f"     ❌ Failed to ingest chunk {i+1}")

                except Exception as e:
                    print(f"     ❌ Error ingesting chunk {i+1}: {str(e)}")

        print(f"\n✅ Indexing completed!")
        print(f"   Total chunks processed: {total_chunks}")
        print(f"   Successful ingests: {successful_ingests}")

        if successful_ingests > 0:
            print(f"   🎉 Your textbook content is now indexed and ready for the Robotics Assistant!")
            return True
        else:
            print(f"   ❌ No content was successfully indexed")
            return False

    except Exception as e:
        print(f"❌ Error during indexing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("📚 Indexing Physical AI & Humanoid Robotics Textbook Content\n")

    success = index_content_to_rag()

    if success:
        print(f"\n🎉 Content indexing completed successfully!")
        print(f"   Your Robotics Assistant chatbot now has access to the full textbook content.")
        print(f"   It can answer questions based on all modules: ROS 2, Digital Twin, AI-Robot Brain, and VLA systems.")
    else:
        print(f"\n❌ Content indexing failed.")
        print(f"   Make sure your RAG API server is running before executing this script.")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)