"""
Document parser for the Physical AI & Humanoid Robotics Book RAG system
Parses MDX files and extracts content for vector storage
"""
import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import frontmatter  # For parsing markdown with frontmatter
import markdown
from bs4 import BeautifulSoup
import asyncio
from slugify import slugify
import logging

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a chunk of document content with metadata"""
    def __init__(self, content: str, chapter_id: str, chapter_title: str,
                 section_title: Optional[str] = None, metadata: Optional[Dict] = None):
        self.content = content
        self.chapter_id = chapter_id
        self.chapter_title = chapter_title
        self.section_title = section_title
        self.metadata = metadata or {}


class MDXParser:
    """Parses MDX files and extracts content for the RAG system"""

    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def parse_mdx_file(self, file_path: str) -> List[DocumentChunk]:
        """
        Parse an MDX file and extract content chunks with metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract frontmatter if present
            try:
                post = frontmatter.loads(content)
                md_content = post.content
                metadata = post.metadata
            except:
                # If no frontmatter, treat entire content as markdown
                md_content = content
                metadata = {}

            # Extract chapter info from file path
            path_parts = Path(file_path).parts
            chapter_id = self._extract_chapter_id(file_path)
            chapter_title = self._extract_chapter_title(md_content, metadata)

            # Convert MDX to plain text (remove JSX components, keep content)
            plain_text = self._convert_mdx_to_text(md_content)

            # Split content into semantic chunks
            chunks = self._semantic_chunk(plain_text, chapter_id, chapter_title, metadata)

            return chunks

        except Exception as e:
            logger.error(f"Error parsing MDX file {file_path}: {e}")
            return []

    def _extract_chapter_id(self, file_path: str) -> str:
        """Extract chapter ID from file path"""
        path = Path(file_path)
        # Format: docs/module-1-ros/chapter-1-intro.md -> module-1-ros-chapter-1-intro
        parts = path.parts
        # Find the index of 'docs' and take the next two parts
        try:
            docs_idx = parts.index('docs')
            module_part = parts[docs_idx + 1] if docs_idx + 1 < len(parts) else 'unknown'
            file_name = path.stem
            return f"{module_part}-{file_name}"
        except ValueError:
            # If 'docs' not in path, use the file name
            return slugify(path.stem)

    def _extract_chapter_title(self, content: str, metadata: Dict) -> str:
        """Extract chapter title from metadata or content"""
        # Try to get title from metadata first
        if metadata and 'title' in metadata:
            return metadata['title']

        # If not in metadata, try to extract from content
        # Look for first heading
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):  # H1 heading
                return line[2:].strip()  # Remove '# ' and whitespace

        # If no H1 heading, create title from first 50 chars
        first_fifty = content[:50].strip()
        if len(content) > 50:
            first_fifty += "..."
        return first_fifty

    def _convert_mdx_to_text(self, mdx_content: str) -> str:
        """
        Convert MDX content to plain text, preserving important content
        while removing JSX components and complex formatting
        """
        # First, remove JSX components (anything between {} that looks like JSX)
        # This is a simplified approach - for a full implementation, you'd need a proper MDX parser
        jsx_pattern = r'\{[^{}]*\}'  # Basic pattern for JSX expressions
        text_content = re.sub(jsx_pattern, '', mdx_content)

        # Convert markdown to HTML then to plain text to remove markdown formatting
        html_content = markdown.markdown(text_content)
        soup = BeautifulSoup(html_content, 'html.parser')
        plain_text = soup.get_text(separator=' ')

        # Clean up extra whitespace
        plain_text = re.sub(r'\s+', ' ', plain_text).strip()

        return plain_text

    def _semantic_chunk(self, content: str, chapter_id: str, chapter_title: str,
                       metadata: Dict) -> List[DocumentChunk]:
        """
        Split content into semantic chunks based on headings and natural breaks
        """
        chunks = []

        # Split by headings (H2, H3) to maintain semantic boundaries
        heading_pattern = r'\n(##\s.*?\n|###\s.*?\n)'
        parts = re.split(heading_pattern, content)

        current_section = ""

        for i, part in enumerate(parts):
            # Check if this part is a heading
            if re.match(r'##\s|###\s', part.strip()):
                # If we have accumulated content, save it as a chunk
                if current_section.strip():
                    subchunks = self._break_large_section(current_section, chapter_id,
                                                        chapter_title, metadata)
                    chunks.extend(subchunks)

                # Start new section with the heading
                current_section = part
            else:
                # Add content to current section
                current_section += part

        # Don't forget the last section
        if current_section.strip():
            subchunks = self._break_large_section(current_section, chapter_id,
                                                chapter_title, metadata)
            chunks.extend(subchunks)

        return chunks

    def _break_large_section(self, section: str, chapter_id: str, chapter_title: str,
                           metadata: Dict) -> List[DocumentChunk]:
        """
        Break a large section into smaller chunks if needed
        """
        chunks = []

        # If the section is already small enough, return as is
        if len(section) <= self.max_chunk_size:
            section_title = self._extract_section_title(section)
            return [DocumentChunk(section, chapter_id, chapter_title, section_title, metadata)]

        # Otherwise, break into smaller chunks
        sentences = re.split(r'[.!?]+\s+', section)
        current_chunk = ""

        for sentence in sentences:
            # Check if adding this sentence would exceed the chunk size
            if len(current_chunk) + len(sentence) > self.max_chunk_size:
                # If the current chunk is substantial, save it
                if len(current_chunk) > 100:  # Minimum chunk size
                    section_title = self._extract_section_title(current_chunk)
                    chunks.append(DocumentChunk(current_chunk.strip(), chapter_id,
                                              chapter_title, section_title, metadata))

                # Start a new chunk with overlap from the previous one
                if self.overlap > 0 and len(current_chunk) > self.overlap:
                    current_chunk = current_chunk[-self.overlap:] + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence

        # Add the final chunk if it has content
        if current_chunk.strip() and len(current_chunk) > 50:  # Minimum chunk size
            section_title = self._extract_section_title(current_chunk)
            chunks.append(DocumentChunk(current_chunk.strip(), chapter_id,
                                      chapter_title, section_title, metadata))

        return chunks

    def _extract_section_title(self, content: str) -> Optional[str]:
        """
        Extract the title of a section (first H2 or H3 if available)
        """
        lines = content.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            if line.startswith('## ') or line.startswith('### '):
                # Remove the heading marker and return the title
                return line.split(' ', 1)[1].strip()
        return None


class BookIngestor:
    """Main class for ingesting book content into the RAG system"""

    def __init__(self, source_dir: str, parser: MDXParser):
        self.source_dir = Path(source_dir)
        self.parser = parser
        self.logger = logging.getLogger(__name__)

    async def ingest_book(self) -> List[DocumentChunk]:
        """
        Ingest all MDX files from the source directory
        """
        all_chunks = []

        # Find all MDX and MD files in the source directory
        mdx_files = list(self.source_dir.rglob("*.mdx")) + list(self.source_dir.rglob("*.md"))

        self.logger.info(f"Found {len(mdx_files)} files to process")

        for file_path in mdx_files:
            self.logger.info(f"Processing: {file_path}")
            chunks = self.parser.parse_mdx_file(str(file_path))
            all_chunks.extend(chunks)
            self.logger.info(f"Extracted {len(chunks)} chunks from {file_path.name}")

        self.logger.info(f"Total chunks extracted: {len(all_chunks)}")
        return all_chunks

    def process_file(self, file_path: str) -> List[DocumentChunk]:
        """
        Process a single file and return its chunks
        """
        return self.parser.parse_mdx_file(file_path)


def main():
    """
    Main function to run the parser on the book content
    """
    import argparse

    parser = argparse.ArgumentParser(description='Parse MDX files for RAG system')
    parser.add_argument('--source-dir', type=str, required=True,
                       help='Directory containing MDX files')
    parser.add_argument('--output-file', type=str, default='chunks.json',
                       help='Output file for chunks (JSON format)')
    parser.add_argument('--max-chunk-size', type=int, default=1000,
                       help='Maximum size of each chunk')
    parser.add_argument('--overlap', type=int, default=100,
                       help='Overlap between chunks')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create parser
    mdx_parser = MDXParser(max_chunk_size=args.max_chunk_size, overlap=args.overlap)

    # Create ingestor
    ingestor = BookIngestor(args.source_dir, mdx_parser)

    # Run ingestion
    import asyncio
    chunks = asyncio.run(ingestor.ingest_book())

    # Save chunks to JSON file
    import json
    chunk_data = []
    for chunk in chunks:
        chunk_data.append({
            'content': chunk.content,
            'chapter_id': chunk.chapter_id,
            'chapter_title': chunk.chapter_title,
            'section_title': chunk.section_title,
            'metadata': chunk.metadata
        })

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(chunk_data, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(chunks)} chunks and saved to {args.output_file}")


if __name__ == "__main__":
    main()