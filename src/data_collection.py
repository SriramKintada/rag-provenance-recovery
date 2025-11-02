"""
Data Collection Module for RAG Document Provenance Recovery.

This module handles fetching documents from multiple sources:
- ArXiv papers (via web search + PDF extraction)
- Wikipedia articles (via web fetch + HTML parsing)
"""

import os
import re
import json
import logging
from typing import List, Dict
from io import BytesIO

import requests
from bs4 import BeautifulSoup

# PDF extraction
try:
    from pypdf import PdfReader
except ImportError:
    # Fallback for older versions
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("Warning: PDF reading library not available. Install pypdf or PyPDF2")
        PdfReader = None

from src.utils import setup_logging, retry_with_backoff


class DocumentCollector:
    """
    Manages document collection from multiple sources.

    Attributes:
        output_dir (str): Directory to save raw documents
        metadata_file (str): Path to metadata JSON
        log_file (str): Path to log file
        collected_docs (list): List of collected document dicts
        logger: Logging instance
    """

    def __init__(
        self,
        output_dir: str = "data/raw",
        metadata_file: str = "data/metadata.json",
        log_file: str = "logs/data_collection.log"
    ):
        """
        Initialize DocumentCollector.

        Args:
            output_dir: Directory to save raw documents
            metadata_file: Path to metadata JSON
            log_file: Path to log file
        """
        self.output_dir = output_dir
        self.metadata_file = metadata_file
        self.log_file = log_file
        self.collected_docs = []

        # Setup logging
        self.logger = setup_logging("DocumentCollector", log_file)

        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        self.logger.info("DocumentCollector initialized")
        self.logger.info(f"  Output directory: {output_dir}")
        self.logger.info(f"  Metadata file: {metadata_file}")

    def search_arxiv_papers(
        self,
        queries: List[str],
        max_papers_per_query: int = 3
    ) -> List[Dict]:
        """
        Search ArXiv for papers.

        Args:
            queries: List of search queries
            max_papers_per_query: Maximum papers to fetch per query

        Returns:
            List of paper metadata dicts with arxiv_id, title, url, pdf_url
        """
        papers = []
        arxiv_pattern = r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)'

        self.logger.info(f"Searching ArXiv with {len(queries)} queries")

        for query in queries:
            self.logger.info(f"  Query: '{query}'")

            try:
                # Search using requests to arxiv.org/search
                search_url = f"https://arxiv.org/search/?query={query.replace(' ', '+')}&searchtype=all"

                response = requests.get(search_url, timeout=30)
                response.raise_for_status()

                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find paper entries
                found_count = 0
                for result in soup.find_all('li', class_='arxiv-result'):
                    if found_count >= max_papers_per_query:
                        break

                    # Extract arXiv ID
                    link = result.find('p', class_='list-title')
                    if link and link.find('a'):
                        href = link.find('a')['href']
                        match = re.search(arxiv_pattern, href)

                        if match:
                            arxiv_id = match.group(1)

                            # Extract title
                            title_elem = result.find('p', class_='title')
                            title = title_elem.text.strip() if title_elem else f'ArXiv Paper {arxiv_id}'

                            paper = {
                                'arxiv_id': arxiv_id,
                                'title': title,
                                'url': f'https://arxiv.org/abs/{arxiv_id}',
                                'pdf_url': f'https://arxiv.org/pdf/{arxiv_id}.pdf',
                                'query': query,
                                'source': 'arxiv'
                            }

                            papers.append(paper)
                            found_count += 1

                            self.logger.info(f"    Found: {arxiv_id} - {title[:50]}...")

                self.logger.info(f"  Total found for '{query}': {found_count}")

            except Exception as e:
                self.logger.error(f"  Error searching for '{query}': {e}")

        self.logger.info(f"TOTAL ArXiv papers found: {len(papers)}")
        if len(papers) > 0:
            self.logger.info(f"  Sample: {papers[0]['title']}")

        return papers

    def fetch_pdf_text(
        self,
        pdf_url: str,
        timeout: int = 30
    ) -> str:
        """
        Download PDF and extract text.

        Args:
            pdf_url: Direct URL to PDF file
            timeout: Request timeout in seconds

        Returns:
            Extracted text as string

        Raises:
            requests.RequestException: If download fails
            Exception: If PDF parsing fails
        """
        self.logger.info(f"Fetching PDF: {pdf_url}")

        try:
            # Download PDF
            response = requests.get(pdf_url, timeout=timeout)
            response.raise_for_status()

            pdf_size = len(response.content)
            self.logger.info(f"  Downloaded: {pdf_size / 1024:.1f} KB")

            # Parse PDF
            if PdfReader is None:
                raise ImportError("PDF reader not available")

            pdf = PdfReader(BytesIO(response.content))
            num_pages = len(pdf.pages)

            # Extract text from all pages
            text = "\n".join(
                page.extract_text()
                for page in pdf.pages
            )

            # Validate
            if len(text) < 100:
                raise ValueError(f"Extracted text too short: {len(text)} chars")

            self.logger.info(f"  Extracted: {len(text)} chars from {num_pages} pages")

            return text

        except requests.exceptions.RequestException as e:
            self.logger.error(f"  Download failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"  PDF parsing failed: {e}")
            raise

    def search_wikipedia_articles(
        self,
        topics: List[str],
        max_articles: int = 5
    ) -> List[Dict]:
        """
        Fetch Wikipedia articles.

        Args:
            topics: List of Wikipedia topics
            max_articles: Maximum articles to fetch

        Returns:
            List of article dicts with title, url, text, topic
        """
        articles = []

        self.logger.info(f"Fetching Wikipedia articles for {len(topics[:max_articles])} topics")

        for topic in topics[:max_articles]:
            self.logger.info(f"  Topic: '{topic}'")

            try:
                # Construct Wikipedia URL
                topic_url = topic.replace(' ', '_')
                url = f"https://en.wikipedia.org/wiki/{topic_url}"

                # Fetch HTML with user agent header (Wikipedia blocks basic requests)
                headers = {
                    'User-Agent': 'RAG-Research-Bot/1.0 (Educational Project; Python/requests)'
                }
                response = requests.get(url, timeout=30, headers=headers)
                response.raise_for_status()

                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract main content
                content_div = soup.find('div', {'id': 'mw-content-text'})
                if not content_div:
                    self.logger.warning(f"  No content found for '{topic}'")
                    continue

                # Clean text
                for tag in content_div.find_all(['script', 'style', 'sup', 'table']):
                    tag.decompose()

                text = content_div.get_text(separator='\n', strip=True)

                # Remove [edit] and reference markers
                text = re.sub(r'\[edit\]', '', text)
                text = re.sub(r'\[\d+\]', '', text)

                # Validate
                if len(text) < 1000:
                    self.logger.warning(f"  Article too short: {len(text)} chars")
                    continue

                article = {
                    'title': topic,
                    'url': url,
                    'text': text,
                    'topic': topic,
                    'source': 'wikipedia'
                }

                articles.append(article)
                self.logger.info(f"  Fetched: {len(text)} chars")

            except Exception as e:
                self.logger.error(f"  Failed to fetch '{topic}': {e}")

        self.logger.info(f"TOTAL Wikipedia articles: {len(articles)}")

        return articles

    def collect_all_documents(
        self,
        arxiv_queries: List[str],
        wikipedia_topics: List[str]
    ) -> List[Dict]:
        """
        Orchestrate collection from all sources.

        Args:
            arxiv_queries: List of ArXiv search queries
            wikipedia_topics: List of Wikipedia topics

        Returns:
            List of document dicts with doc_id, title, url, source, text
        """
        self.logger.info("="*60)
        self.logger.info("STARTING DOCUMENT COLLECTION")
        self.logger.info("="*60)

        all_docs = []

        # 1. ArXiv papers
        self.logger.info("\n[1/2] Collecting ArXiv papers...")
        papers = self.search_arxiv_papers(arxiv_queries)

        for i, paper in enumerate(papers):
            try:
                text = self.fetch_pdf_text(paper['pdf_url'])

                doc = {
                    'doc_id': f"arxiv_{i}",
                    'title': paper['title'],
                    'url': paper['url'],
                    'source': 'arxiv',
                    'text': text,
                    'query': paper['query']
                }

                all_docs.append(doc)
                self.collected_docs.append(doc)

            except Exception as e:
                self.logger.error(f"Skipping paper {paper.get('arxiv_id', 'unknown')}: {e}")

        # 2. Wikipedia articles
        self.logger.info("\n[2/2] Collecting Wikipedia articles...")
        articles = self.search_wikipedia_articles(wikipedia_topics)

        for i, article in enumerate(articles):
            doc = {
                'doc_id': f"wikipedia_{i}",
                'title': article['title'],
                'url': article['url'],
                'source': 'wikipedia',
                'text': article['text'],
                'topic': article['topic']
            }

            all_docs.append(doc)
            self.collected_docs.append(doc)

        # Final verification
        self.logger.info("\n" + "="*60)
        self.logger.info(f"COLLECTION COMPLETE: {len(all_docs)} documents")
        self.logger.info(f"  ArXiv papers: {sum(1 for d in all_docs if d['source'] == 'arxiv')}")
        self.logger.info(f"  Wikipedia: {sum(1 for d in all_docs if d['source'] == 'wikipedia')}")
        self.logger.info("="*60)

        return all_docs

    def save_documents(self, documents: List[Dict]) -> None:
        """
        Save documents to disk with metadata.

        Args:
            documents: List of document dicts

        Verification: Checks file existence and sizes
        """
        self.logger.info(f"Saving {len(documents)} documents to {self.output_dir}")

        # Save individual text files
        for doc in documents:
            filename = f"{doc['doc_id']}.txt"
            filepath = os.path.join(self.output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(doc['text'])

            # Verify file exists
            assert os.path.exists(filepath), f"File not created: {filepath}"
            file_size = os.path.getsize(filepath)
            self.logger.info(f"  Saved: {filename} ({file_size / 1024:.1f} KB)")

        # Save metadata
        metadata = [
            {
                'doc_id': d['doc_id'],
                'title': d['title'],
                'url': d['url'],
                'source': d['source'],
                'text_length': len(d['text']),
                'file_path': os.path.join(self.output_dir, f"{d['doc_id']}.txt")
            }
            for d in documents
        ]

        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        # Verify metadata file
        assert os.path.exists(self.metadata_file), "Metadata file not created"
        self.logger.info(f"Metadata saved: {self.metadata_file}")

        # Final check
        files = os.listdir(self.output_dir)
        self.logger.info(f"[OK] VERIFIED: {len(files)} files in {self.output_dir}")
