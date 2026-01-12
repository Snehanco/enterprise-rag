# text_cleaner.py
import re
from typing import List
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


class TextCleaner:
    """
    Text cleaning for RAG ingestion pipeline using Presidio for PII detection.
    Presidio automatically detects and anonymizes various PII types.
    """

    def __init__(self):
        """
        Initialize TextCleaner with Presidio engines.

        Presidio detects:
        - EMAIL_ADDRESS
        - PHONE_NUMBER
        - PERSON
        - LOCATION
        - ORGANIZATION
        - CREDIT_CARD
        - IBAN_CODE
        - MEDICAL_LICENSE
        - US_PASSPORT
        - US_SSN
        - IP_ADDRESS
        And more...
        """
        logger.info("Initializing TextCleaner with Presidio")
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        # Violent/profane words to redact
        self.violent_words = [
            "kill",
            "murder",
            "assassinate",
            "shoot",
            "stab",
            "rape",
            "assault",
            "torture",
            "massacre",
            "bomb",
            "violence",
            "violent",
            "attack",
            "destroy",
            "harm",
            "hurt",
            "wound",
            "maim",
            "execution",
        ]

        logger.info("TextCleaner initialized successfully")

    def remove_pii(self, text: str, language: str = "en") -> str:
        """
        Remove PII using Presidio analyzer and anonymizer.

        Automatically detects and anonymizes:
        - Email addresses → [EMAIL_ADDRESS]
        - Phone numbers → [PHONE_NUMBER]
        - Person names → [PERSON]
        - Locations → [LOCATION]
        - Organizations → [ORGANIZATION]
        - Credit card numbers → [CREDIT_CARD]
        - IBAN codes → [IBAN_CODE]
        - Passport numbers → [US_PASSPORT]
        - SSN → [US_SSN]
        - IP addresses → [IP_ADDRESS]
        - And more...

        Args:
            text: Text to anonymize
            language: Language code (default: "en")

        Returns:
            Anonymized text
        """
        try:
            # Analyze text for PII
            results = self.analyzer.analyze(text=text, language=language)

            if not results:
                logger.debug("No PII detected in text")
                return text

            # Filter out DATE_TIME entities to preserve dates/times in text
            results = [
                result for result in results if result.entity_type != "DATE_TIME"
            ]

            # Log detected PII for monitoring
            pii_types = set(result.entity_type for result in results)
            logger.debug(f"Detected PII types: {pii_types}")

            # Anonymize detected PII
            anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
            return anonymized.text

        except Exception as e:
            logger.error(f"Error during PII removal: {str(e)}")
            return text

    def remove_violent_words(
        self, text: str, redaction_token: str = "[REDACTED]"
    ) -> str:
        """
        Remove violent/profane words.

        Args:
            text: Text to clean
            redaction_token: Token to replace violent words with

        Returns:
            Text with violent words redacted
        """
        for word in self.violent_words:
            pattern = r"\b" + re.escape(word) + r"\b"
            text = re.sub(pattern, redaction_token, text, flags=re.IGNORECASE)

        return text

    def remove_urls(self, text: str) -> str:
        """
        Remove HTTP/HTTPS and www URLs.

        Args:
            text: Text to clean

        Returns:
            Text without URLs
        """
        # Remove http/https URLs
        text = re.sub(r"https?://\S+", "", text)

        # Remove www URLs
        text = re.sub(r"www\.\S+", "", text)

        return text

    def remove_html_tags(self, text: str) -> str:
        """
        Remove HTML/XML tags.

        Args:
            text: Text to clean

        Returns:
            Text without HTML tags
        """
        return re.sub(r"<[^>]+>", "", text)

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace, quotes, and control characters.

        Operations:
        - Normalize smart quotes to standard quotes
        - Remove control characters (except newlines/tabs)
        - Normalize multiple spaces to single space
        - Normalize multiple newlines to double newline
        - Remove trailing whitespace from lines

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        # Normalize smart quotes to standard quotes
        text = re.sub(r"[\u201c\u201d]", '"', text)  # Curly double quotes
        text = re.sub(r"[\u2018\u2019]", "'", text)  # Curly single quotes

        # Remove control characters (keep newlines, tabs, carriage returns)
        text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t\r")

        # Normalize multiple spaces to single space
        text = re.sub(r" {2,}", " ", text)

        # Normalize multiple newlines to double newline (preserve paragraphs)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove trailing whitespace from each line
        lines = text.split("\n")
        lines = [line.rstrip() for line in lines]
        text = "\n".join(lines)

        return text.strip()

    def clean_text(self, text: str, language: str = "en") -> str:
        """
        Complete text cleaning pipeline.

        Operations in order:
        1. Remove HTML tags
        2. Remove URLs
        3. Remove PII (using Presidio)
        4. Remove violent words
        5. Normalize whitespace

        Args:
            text: Raw text string
            language: Language code for Presidio analysis (default: "en")

        Returns:
            Cleaned text string
        """
        logger.debug("Starting text cleaning pipeline")

        # Step 1: Remove HTML
        text = self.remove_html_tags(text)
        logger.debug("Removed HTML tags")

        # Step 2: Remove URLs
        text = self.remove_urls(text)
        logger.debug("Removed URLs")

        # Step 3: Remove PII with Presidio
        text = self.remove_pii(text, language=language)
        logger.debug("Removed PII using Presidio")

        # Step 4: Remove violent words
        text = self.remove_violent_words(text)
        logger.debug("Removed violent words")

        # Step 5: Normalize whitespace
        text = self.normalize_whitespace(text)
        logger.debug("Normalized whitespace")

        logger.debug("Text cleaning completed")
        return text

    def clean_document(self, doc: Document, language: str = "en") -> Document:
        """
        Clean a single Document object.

        Args:
            doc: LangChain Document object
            language: Language code for Presidio analysis

        Returns:
            Document with cleaned page_content
        """
        doc.page_content = self.clean_text(doc.page_content, language=language)
        return doc

    def clean_documents(
        self, documents: List[Document], language: str = "en"
    ) -> List[Document]:
        """
        Clean a list of Document objects.

        Args:
            documents: List of LangChain Document objects
            language: Language code for Presidio analysis (default: "en")

        Returns:
            List of cleaned Document objects
        """
        logger.info(f"Starting to clean {len(documents)} documents")

        cleaned_docs = []
        for i, doc in enumerate(documents, 1):
            logger.info(f"Cleaning document {i}/{len(documents)}")
            cleaned_doc = self.clean_document(doc, language=language)
            cleaned_docs.append(cleaned_doc)

        logger.info(f"Successfully cleaned {len(cleaned_docs)} documents")
        return cleaned_docs

    def get_pii_stats(self, text: str, language: str = "en") -> dict:
        """
        Analyze text and return statistics about detected PII.

        Args:
            text: Text to analyze
            language: Language code for analysis

        Returns:
            Dictionary with PII statistics
        """
        try:
            results = self.analyzer.analyze(text=text, languages=language)

            # Count PII by type
            pii_counts = {}
            for result in results:
                pii_type = result.entity_type
                pii_counts[pii_type] = pii_counts.get(pii_type, 0) + 1

            return {
                "total_pii_found": len(results),
                "pii_types": pii_counts,
                "text_length": len(text),
            }

        except Exception as e:
            logger.error(f"Error analyzing PII: {str(e)}")
            return {"error": str(e)}
