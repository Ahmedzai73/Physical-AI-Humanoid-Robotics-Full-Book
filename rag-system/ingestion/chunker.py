"""
Content chunker for the Physical AI & Humanoid Robotics Book RAG system
Responsible for breaking down content into semantic chunks for vector storage
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class ContentChunk:
    """Represents a chunk of content with metadata"""
    id: str
    content: str
    chapter_id: str
    chapter_title: str
    section_title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    similarity_score: Optional[float] = None
    position: int = 0


class SemanticChunker:
    """Chunks content based on semantic boundaries and meaning"""

    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100,
                 overlap: int = 100, sentence_buffer: int = 3):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        self.sentence_buffer = sentence_buffer  # Number of sentences to buffer at boundaries
        self.sentence_splitter = re.compile(r'[.!?]+\s+')

    def chunk_by_semantic_boundaries(self, content: str, chapter_id: str,
                                   chapter_title: str, metadata: Optional[Dict] = None) -> List[ContentChunk]:
        """
        Chunk content based on semantic boundaries like headings, paragraphs, etc.
        """
        chunks = []

        # Split by semantic boundaries (headings, paragraphs)
        semantic_parts = self._split_by_semantic_boundaries(content)

        chunk_id = 0
        for part in semantic_parts:
            if len(part.strip()) < self.min_chunk_size:
                continue

            # If part is too large, further chunk it
            if len(part) > self.max_chunk_size:
                sub_chunks = self._chunk_large_part(part, chapter_id, chapter_title, metadata)
                for sub_chunk in sub_chunks:
                    sub_chunk.id = f"{chapter_id}_chunk_{chunk_id}"
                    sub_chunk.position = chunk_id
                    chunks.append(sub_chunk)
                    chunk_id += 1
            else:
                chunk = ContentChunk(
                    id=f"{chapter_id}_chunk_{chunk_id}",
                    content=part,
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    metadata=metadata,
                    position=chunk_id
                )
                chunks.append(chunk)
                chunk_id += 1

        return chunks

    def _split_by_semantic_boundaries(self, content: str) -> List[str]:
        """
        Split content by semantic boundaries like headings, paragraphs, etc.
        """
        # First split by headings (H2 and H3)
        heading_pattern = r'\n(##\s.*?\n|###\s.*?\n)'
        parts = re.split(heading_pattern, content)

        # Reassemble parts with headings
        reassembled_parts = []
        i = 0
        while i < len(parts):
            part = parts[i]
            # Check if next part is a heading
            if i + 1 < len(parts) and re.match(r'##\s|###\s', parts[i + 1].strip()):
                # Combine content with its heading
                part += parts[i + 1]
                i += 2
            else:
                i += 1
            reassembled_parts.append(part)

        # Now split by paragraphs and other semantic boundaries
        final_parts = []
        for part in reassembled_parts:
            if len(part) <= self.max_chunk_size:
                final_parts.append(part)
            else:
                # Further split large parts by paragraphs
                paragraphs = part.split('\n\n')
                current_chunk = ""

                for paragraph in paragraphs:
                    if len(current_chunk) + len(paragraph) <= self.max_chunk_size:
                        current_chunk += f"\n\n{paragraph}"
                    else:
                        if current_chunk.strip():
                            final_parts.append(current_chunk.strip())
                        current_chunk = paragraph

                if current_chunk.strip():
                    final_parts.append(current_chunk.strip())

        return [part for part in final_parts if part.strip()]

    def _chunk_large_part(self, part: str, chapter_id: str, chapter_title: str,
                         metadata: Optional[Dict]) -> List[ContentChunk]:
        """
        Further chunk a large part of content into smaller pieces
        """
        chunks = []

        # Split into sentences
        sentences = self.sentence_splitter.split(part)
        current_chunk = ""
        chunk_id = 0

        for sentence in sentences:
            # Check if adding this sentence would exceed the chunk size
            if len(current_chunk) + len(sentence) > self.max_chunk_size:
                # If the current chunk is substantial, save it
                if len(current_chunk) >= self.min_chunk_size:
                    chunk = ContentChunk(
                        id=f"{chapter_id}_chunk_{len(chunks)}",
                        content=current_chunk.strip(),
                        chapter_id=chapter_id,
                        chapter_title=chapter_title,
                        metadata=metadata,
                        position=len(chunks)
                    )
                    chunks.append(chunk)

                # Start a new chunk with overlap from the previous one
                if self.overlap > 0 and len(current_chunk) > self.overlap:
                    # Get the last few sentences for overlap
                    overlap_content = self._get_overlap_content(current_chunk, self.overlap)
                    current_chunk = overlap_content + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence

        # Add the final chunk if it has content
        if current_chunk.strip() and len(current_chunk) >= self.min_chunk_size:
            chunk = ContentChunk(
                id=f"{chapter_id}_chunk_{len(chunks)}",
                content=current_chunk.strip(),
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                metadata=metadata,
                position=len(chunks)
            )
            chunks.append(chunk)

        return chunks

    def _get_overlap_content(self, text: str, overlap_size: int) -> str:
        """
        Get the last overlap_size characters from text, trying to break at sentence boundaries
        """
        if len(text) <= overlap_size:
            return text

        # Try to find a sentence boundary near the overlap point
        start_pos = len(text) - overlap_size
        for i in range(start_pos, len(text)):
            if text[i] in '.!?':
                # Include the punctuation and any following whitespace
                end_pos = i + 1
                while end_pos < len(text) and text[end_pos].isspace():
                    end_pos += 1
                return text[end_pos:]

        # If no sentence boundary found, just return the last overlap_size chars
        return text[-overlap_size:]

    def chunk_by_topic(self, content: str, chapter_id: str, chapter_title: str,
                      metadata: Optional[Dict] = None) -> List[ContentChunk]:
        """
        Chunk content based on topic coherence using NLP techniques
        """
        # This is a simplified implementation - in a full implementation,
        # you would use more sophisticated NLP techniques like:
        # - Topic modeling (LDA, BERTopic)
        # - Sentence embeddings and clustering
        # - TextTiling algorithm
        # For now, we'll use the semantic boundary method
        return self.chunk_by_semantic_boundaries(content, chapter_id, chapter_title, metadata)


class ContentProcessor:
    """Main processor for content chunking and preparation"""

    def __init__(self, chunker: SemanticChunker):
        self.chunker = chunker
        self.logger = logging.getLogger(__name__)

    def process_content(self, content: str, chapter_id: str, chapter_title: str,
                       metadata: Optional[Dict] = None) -> List[ContentChunk]:
        """
        Process content and return properly chunked content
        """
        self.logger.info(f"Processing content for chapter: {chapter_id}")

        # Chunk the content
        chunks = self.chunker.chunk_by_semantic_boundaries(
            content, chapter_id, chapter_title, metadata
        )

        self.logger.info(f"Created {len(chunks)} chunks for chapter: {chapter_id}")

        return chunks

    def merge_similar_chunks(self, chunks: List[ContentChunk], similarity_threshold: float = 0.8) -> List[ContentChunk]:
        """
        Merge chunks that are highly similar to reduce redundancy
        """
        if not chunks:
            return chunks

        # For this implementation, we'll use a simple approach
        # In a full implementation, you would calculate embeddings and use similarity measures
        merged_chunks = [chunks[0]]

        for current_chunk in chunks[1:]:
            last_chunk = merged_chunks[-1]

            # Simple similarity check based on content overlap
            similarity = self._calculate_content_similarity(last_chunk.content, current_chunk.content)

            if similarity > similarity_threshold:
                # Merge the chunks
                merged_content = last_chunk.content + " " + current_chunk.content
                merged_chunk = ContentChunk(
                    id=last_chunk.id,
                    content=merged_content,
                    chapter_id=last_chunk.chapter_id,
                    chapter_title=last_chunk.chapter_title,
                    section_title=last_chunk.section_title,
                    metadata=last_chunk.metadata,
                    position=last_chunk.position,
                    similarity_score=similarity
                )
                merged_chunks[-1] = merged_chunk
            else:
                merged_chunks.append(current_chunk)

        return merged_chunks

    def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text chunks (simplified implementation)
        """
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 0.0

        return len(intersection) / len(union)


def main():
    """
    Main function to demonstrate the chunker functionality
    """
    import argparse

    parser = argparse.ArgumentParser(description='Chunk content for RAG system')
    parser.add_argument('--max-chunk-size', type=int, default=1000,
                       help='Maximum size of each chunk')
    parser.add_argument('--min-chunk-size', type=int, default=100,
                       help='Minimum size of each chunk')
    parser.add_argument('--overlap', type=int, default=100,
                       help='Overlap between chunks')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create chunker
    chunker = SemanticChunker(
        max_chunk_size=args.max_chunk_size,
        min_chunk_size=args.min_chunk_size,
        overlap=args.overlap
    )

    # Create processor
    processor = ContentProcessor(chunker)

    # Example usage (would typically read from files)
    example_content = """
# Introduction to ROS 2

## What is ROS 2?

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software.
It's a collection of tools, libraries, and conventions that aim to simplify the task
of creating complex and robust robot behavior across a wide variety of robot platforms.

## Architecture Overview

ROS 2 uses a DDS (Data Distribution Service) based architecture which provides a
publish-subscribe communication model. This allows for better real-time performance
and improved security compared to ROS 1.

### Nodes

Nodes are the fundamental units of computation in ROS 2. Each node is a process that
performs computation. Nodes written in different programming languages can be
integrated into the same system.

### Topics and Messages

Topics are named buses over which nodes exchange messages. Messages are the data
that flows between nodes. The publish-subscribe communication pattern allows for
loose coupling between nodes.

## Installation

To install ROS 2, you'll need to follow the official installation guide for your
operating system. Make sure to source the setup script in your terminal before
using ROS 2 commands.

This section continues with more detailed information about ROS 2 concepts,
including services, actions, parameters, and more advanced topics that are
essential for developing robotic applications.

The next paragraph continues with even more content to demonstrate how the
chunker handles longer sections of text that exceed the maximum chunk size.
"""

    # Process the example content
    chunks = processor.process_content(
        example_content,
        chapter_id="module-1-ros-introduction",
        chapter_title="Introduction to ROS 2",
        metadata={"module": "1", "difficulty": "beginner"}
    )

    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} (ID: {chunk.id}, Position: {chunk.position}):")
        print(f"Length: {len(chunk.content)} characters")
        print(f"Content preview: {chunk.content[:100]}...")
        print("-" * 50)

    # Test merging similar chunks
    merged_chunks = processor.merge_similar_chunks(chunks)
    print(f"\nAfter merging similar chunks: {len(merged_chunks)} chunks")


if __name__ == "__main__":
    main()