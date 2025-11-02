"""
Configuration management for RAG Document Provenance Recovery.

Loads environment variables and provides centralized configuration.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Centralized configuration for the entire project"""

    # ============================================================================
    # API CONFIGURATION
    # ============================================================================
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # ============================================================================
    # MODEL CONFIGURATION
    # ============================================================================
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Model specifications
    EMBEDDING_DIMENSIONS = 1536  # For text-embedding-3-small

    # Token limits for models
    TOKEN_LIMITS = {
        "gpt-4o-mini": 128000,  # 128k context window
        "gpt-3.5-turbo": 4096,
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
    }

    # ============================================================================
    # PATH CONFIGURATION
    # ============================================================================
    DATA_DIR = os.getenv("DATA_DIR", "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")

    EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", "embeddings")
    CHROMA_DB_DIR = os.path.join(EMBEDDINGS_DIR, "chroma_db")

    RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
    FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
    TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

    LOGS_DIR = os.getenv("LOGS_DIR", "logs")

    # ============================================================================
    # RAG PARAMETERS
    # ============================================================================
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))

    # Text splitter configuration (from Debugging Playbook)
    TEXT_SPLITTER_SEPARATORS = ['\n\n', '\n', '. ', ' ', '']

    # ============================================================================
    # DATA COLLECTION PARAMETERS
    # ============================================================================
    MAX_PAPERS_PER_ARXIV_QUERY = 3
    MAX_WIKIPEDIA_ARTICLES = 5
    PDF_FETCH_TIMEOUT = 30  # seconds

    # Target corpus size
    MIN_DOCUMENTS = 20
    MAX_DOCUMENTS = 40

    # ============================================================================
    # EVALUATION PARAMETERS
    # ============================================================================
    MAX_QUERIES = int(os.getenv("MAX_QUERIES", "50"))

    # N-gram size for n-gram overlap method
    NGRAM_SIZE = 3

    # Batch size for embedding generation
    EMBEDDING_BATCH_SIZE = 100

    # ============================================================================
    # INVERSE METHOD PARAMETERS
    # ============================================================================
    # For LLM attribution, how many candidate chunks to consider
    LLM_ATTRIBUTION_CANDIDATES = 10

    # TF-IDF parameters
    TFIDF_NGRAM_RANGE = (1, 2)  # Unigrams and bigrams
    TFIDF_MAX_FEATURES = 5000

    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================
    LOG_LEVEL = "INFO"
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    # ============================================================================
    # VALIDATION
    # ============================================================================
    @classmethod
    def validate(cls):
        """
        Validate configuration settings.

        Raises:
            ValueError: If required settings are missing or invalid
        """
        # Check API key
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not set. Please create a .env file with your API key."
            )

        if not cls.OPENAI_API_KEY.startswith('sk-'):
            raise ValueError(
                "Invalid OPENAI_API_KEY format. Key should start with 'sk-'"
            )

        # Validate chunk parameters
        if cls.CHUNK_SIZE <= 0:
            raise ValueError(f"CHUNK_SIZE must be positive, got {cls.CHUNK_SIZE}")

        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({cls.CHUNK_OVERLAP}) must be less than "
                f"CHUNK_SIZE ({cls.CHUNK_SIZE})"
            )

        # Validate retrieval k
        if cls.RETRIEVAL_K <= 0:
            raise ValueError(f"RETRIEVAL_K must be positive, got {cls.RETRIEVAL_K}")

        print("[OK] Configuration validated successfully")
        print(f"  Embedding model: {cls.EMBEDDING_MODEL}")
        print(f"  LLM model: {cls.LLM_MODEL}")
        print(f"  Chunk size: {cls.CHUNK_SIZE} (overlap: {cls.CHUNK_OVERLAP})")
        print(f"  Retrieval k: {cls.RETRIEVAL_K}")

    @classmethod
    def create_directories(cls):
        """Create all required directories if they don't exist"""
        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.CHROMA_DB_DIR,
            cls.FIGURES_DIR,
            cls.TABLES_DIR,
            cls.LOGS_DIR,
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        print("[OK] Directory structure created")

    @classmethod
    def get_token_limit(cls, model=None):
        """
        Get token limit for specified model.

        Args:
            model: Model name (defaults to configured LLM_MODEL)

        Returns:
            int: Token limit for the model
        """
        if model is None:
            model = cls.LLM_MODEL

        return cls.TOKEN_LIMITS.get(model, 4096)  # Default to 4096 if unknown

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"LLM Model: {cls.LLM_MODEL}")
        print(f"Token Limit: {cls.get_token_limit():,}")
        print(f"Chunk Size: {cls.CHUNK_SIZE} (overlap: {cls.CHUNK_OVERLAP})")
        print(f"Retrieval K: {cls.RETRIEVAL_K}")
        print(f"Data Directory: {cls.DATA_DIR}")
        print(f"Embeddings Directory: {cls.EMBEDDINGS_DIR}")
        print(f"Results Directory: {cls.RESULTS_DIR}")
        print(f"Logs Directory: {cls.LOGS_DIR}")
        print("="*60 + "\n")


# Validate configuration on import (will raise error if invalid)
if __name__ == "__main__":
    Config.validate()
    Config.create_directories()
    Config.print_config()
