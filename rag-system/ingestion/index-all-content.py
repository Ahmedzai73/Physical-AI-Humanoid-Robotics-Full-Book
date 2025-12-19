#!/usr/bin/env python3

"""
Index All Content for Physical AI & Humanoid Robotics RAG System

This script indexes all content from the Physical AI & Humanoid Robotics textbook
for the RAG chatbot system.
"""

import os
import sys
import json
from pathlib import Path
import markdown
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import asyncio
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import logging

# Add the rag-system/api directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

from models import DocumentMetadata, QueryRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentIndexer:
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "textbook_content"):
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model for embeddings

        # Create collection if it doesn't exist
        self._create_collection()

    def _create_collection(self):
        """Create Qdrant collection for storing textbook content."""
        try:
            # Check if collection exists
            self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
        except:
            # Create new collection
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "content": {
                        "size": 384,  # Size of all-MiniLM-L6-v2 embeddings
                        "distance": "Cosine"
                    }
                }
            )
            logger.info(f"Created collection '{self.collection_name}'")

    def extract_text_from_markdown(self, md_content: str) -> str:
        """Extract plain text from markdown content."""
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text(separator=' ', strip=True)

    def get_all_md_files(self, root_dir: str) -> List[str]:
        """Recursively find all markdown files in the textbook directory."""
        md_files = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))
        return md_files

    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single document and extract chunks for indexing."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract text from markdown
            text = self.extract_text_from_markdown(content)

            # Create chunks (split by paragraphs or sections)
            chunks = self._create_chunks(text, file_path)

            documents = []
            for i, chunk in enumerate(chunks):
                doc = {
                    'id': f"{file_path.replace('/', '_').replace('\\', '_')}_{i}",
                    'content': chunk,
                    'metadata': {
                        'file_path': file_path,
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                }
                documents.append(doc)

            logger.info(f"Processed {len(chunks)} chunks from {file_path}")
            return documents

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []

    def _create_chunks(self, text: str, file_path: str) -> List[str]:
        """Create chunks of text for indexing."""
        # Split by double newlines (paragraphs) first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        current_chunk = ""
        chunk_size = 0
        max_chunk_size = 512  # tokens/approximate words

        for para in paragraphs:
            # If adding this paragraph would exceed the max size, save current chunk
            if chunk_size + len(para.split()) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
                chunk_size = len(para.split())
            else:
                if current_chunk:
                    current_chunk += " " + para
                else:
                    current_chunk = para
                chunk_size += len(para.split())

                # If even adding a paragraph makes it too big, split it
                if chunk_size > max_chunk_size:
                    # Split the current chunk into smaller parts
                    temp_chunks = self._split_large_chunk(current_chunk, max_chunk_size)
                    chunks.extend(temp_chunks[:-1])  # Add all but the last chunk
                    current_chunk = temp_chunks[-1]  # Keep the remainder
                    chunk_size = len(current_chunk.split())

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())

        # If we still have chunks that are too big, split them
        final_chunks = []
        for chunk in chunks:
            if len(chunk.split()) > max_chunk_size:
                final_chunks.extend(self._split_large_chunk(chunk, max_chunk_size))
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _split_large_chunk(self, text: str, max_size: int) -> List[str]:
        """Split a large chunk of text into smaller pieces."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), max_size):
            chunk_words = words[i:i + max_size]
            chunks.append(' '.join(chunk_words))

        return chunks

    def index_content(self, textbook_dir: str):
        """Index all content from the textbook directory."""
        logger.info(f"Starting to index content from {textbook_dir}")

        # Get all markdown files
        md_files = self.get_all_md_files(textbook_dir)
        logger.info(f"Found {len(md_files)} markdown files to index")

        all_documents = []

        # Process each file
        for file_path in md_files:
            documents = self.process_document(file_path)
            all_documents.extend(documents)

        logger.info(f"Processing {len(all_documents)} document chunks for indexing")

        # Prepare data for Qdrant
        ids = []
        embeddings = []
        payloads = []

        for i, doc in enumerate(all_documents):
            # Generate embedding
            embedding = self.encoder.encode(doc['content']).tolist()

            ids.append(i)
            embeddings.append(embedding)
            payloads.append({
                'content': doc['content'],
                'metadata': doc['metadata']
            })

        # Upload to Qdrant in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    {
                        "id": id_,
                        "vector": {"content": emb},
                        "payload": payload_
                    }
                    for id_, emb, payload_ in zip(batch_ids, batch_embeddings, batch_payloads)
                ]
            )

            logger.info(f"Indexed batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")

        logger.info(f"Successfully indexed {len(all_documents)} document chunks")

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search the indexed content."""
        query_embedding = self.encoder.encode(query).tolist()

        hits = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=("content", query_embedding),
            limit=limit
        )

        results = []
        for hit in hits:
            results.append({
                'content': hit.payload['content'],
                'metadata': hit.payload['metadata'],
                'score': hit.score
            })

        return results


def main():
    """Main function to index all textbook content."""
    # Get the textbook directory (assuming this script is in rag-system/ingestion/)
    textbook_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'docs')
    textbook_dir = os.path.abspath(textbook_dir)

    indexer = ContentIndexer()

    try:
        indexer.index_content(textbook_dir)
        logger.info("Content indexing completed successfully!")

        # Test the search functionality
        test_query = "What is ROS 2?"
        results = indexer.search(test_query)
        logger.info(f"Test search for '{test_query}' returned {len(results)} results")

        if results:
            logger.info(f"Top result: {results[0]['content'][:200]}...")

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()