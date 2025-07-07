import os
import json
import torch
import pickle
import logging
import re
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
import time

# PDF Processing
import PyPDF2
import fitz  # PyMuPDF

# Sentence Transformers & Vector Store
from sentence_transformers import SentenceTransformer
import faiss

# NLTK for text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
@dataclass
class RAGConfig:
    """Central configuration for the RAG system."""
    vector_store_path: str = "./fitness_rag_store_v2"
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Chunking parameters
    chunk_size: int = 300  # Words per chunk
    chunk_overlap_sentences: int = 2 # Number of sentences to overlap

    # Retrieval parameters
    retrieval_k: int = 5
    retrieval_score_threshold: float = 0.2
    max_context_length: int = 2500

DEFAULT_SYSTEM_PROMPT = """Sen, FitTürkAI adında, Bütünsel, Empatik ve Proaktif bir Sağlıklı Yaşam Koçusun.
Aşağıdaki bilgi kaynaklarını kullanarak kullanıcının sorusunu kapsamlı ve anlaşılır bir şekilde yanıtla.

ÖNEMLİ KURALLAR:
- Cevabının başında kesinlikle bir tıp doktoru olmadığını ve verdiğin bilgilerin tıbbi tavsiye niteliği taşımadığını belirt. Herhangi bir sağlık programına başlamadan önce bir doktora danışılması gerektiğini vurgula.
- Yalnızca sana verilen kaynaklardaki bilgileri kullan. Bilgiyi hangi kaynaktan aldığını belirt.
- Eğer kaynaklarda cevap yoksa, "Bu konuda kaynaklarımda kesin bir bilgi bulamadım, ancak genel olarak bilinen şudur ki..." diyerek genel bilgilerini kullanabilirsin.
- Her zaman empatik, motive edici ve cesaret verici bir dil kullan."""

# --- Data Structures ---
@dataclass
class Document:
    """Represents a document chunk with metadata."""
    content: str
    source: str
    doc_type: str  # 'pdf' or 'json'
    chunk_id: str
    metadata: Dict = field(default_factory=dict)

# --- Core Components ---

class TurkishTextProcessor:
    """Handles advanced Turkish text preprocessing, cleaning, and chunking."""

    def __init__(self):
        self.turk_to_ascii_map = str.maketrans('ğüşıöçĞÜŞİÖÇ', 'gusiocGUSIOC')
        # Initialize with basic Turkish stopwords in case download fails
        self.turkish_stopwords = {'ve', 'ile', 'bir', 'bu', 'da', 'de', 'için', 'olan', 'var', 'yok', 'bu', 'şu', 'o', 'ben', 'sen', 'biz', 'siz', 'onlar'}
        self._download_nltk_data()
        # Try to get proper stopwords after download
        try:
            self.turkish_stopwords = set(stopwords.words('turkish'))
        except:
            logger.warning("Could not load Turkish stopwords, using basic set")

    def _download_nltk_data(self):
        """Download required NLTK data with updated tokenizer names."""
        logger.info("Downloading required NLTK data...")

        # Force download the required packages
        try:
            nltk.download('punkt_tab', quiet=True)
            logger.info("Downloaded punkt_tab successfully")
        except Exception as e:
            logger.warning(f"Failed to download punkt_tab: {e}")
            try:
                nltk.download('punkt', quiet=True)
                logger.info("Downloaded punkt successfully")
            except Exception as e2:
                logger.warning(f"Failed to download punkt: {e2}")

        try:
            nltk.download('stopwords', quiet=True)
            logger.info("Downloaded stopwords successfully")
        except Exception as e:
            logger.warning(f"Failed to download stopwords: {e}")
            # Create a minimal set of stopwords if download fails
            self.turkish_stopwords = {'ve', 'ile', 'bir', 'bu', 'da', 'de', 'için', 'olan', 'var', 'yok'}

    def turkish_lower(self, text: str) -> str:
        """Correctly lowercases Turkish text."""
        return text.replace('I', 'ı').replace('İ', 'i').lower()

    def to_ascii(self, text: str) -> str:
        """Converts Turkish characters to their ASCII equivalents."""
        return text.translate(self.turk_to_ascii_map)

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = text.strip()
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\sğüşıöçĞÜŞİÖÇ.,!?-]', '', text)
        return text

    def preprocess_for_embedding(self, text: str) -> str:
        """Prepares text for embedding by cleaning and lowercasing."""
        text = self.clean_text(text)
        text = self.turkish_lower(text)
        return text

    def chunk_text(self, text: str, chunk_size: int, overlap_sentences: int) -> List[str]:
        """Split text into overlapping chunks based on sentences."""
        sentences = []

        # Try multiple tokenization methods
        try:
            sentences = sent_tokenize(text, language='turkish')
        except:
            try:
                sentences = sent_tokenize(text, language='english')
            except:
                # Ultimate fallback: split by punctuation
                logger.warning("NLTK tokenization failed, using basic sentence splitting")
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            # If all tokenization fails, split by length
            words = text.split()
            sentences = []
            for i in range(0, len(words), chunk_size // 2):
                sentences.append(' '.join(words[i:i + chunk_size // 2]))

        chunks = []
        current_chunk_words = []

        for i, sentence in enumerate(sentences):
            sentence_words = sentence.split()
            if len(current_chunk_words) + len(sentence_words) > chunk_size and current_chunk_words:
                chunks.append(" ".join(current_chunk_words))

                overlap_start_index = max(0, i - overlap_sentences)
                overlapped_sentences = sentences[overlap_start_index:i]
                current_chunk_words = " ".join(overlapped_sentences).split()

            current_chunk_words.extend(sentence_words)

        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words))

        return chunks

class PDFProcessor:
    """Handles PDF document processing."""
    def __init__(self, text_processor: TurkishTextProcessor, config: RAGConfig):
        self.text_processor = text_processor
        self.config = config

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF using PyMuPDF with a PyPDF2 fallback."""
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                text = "".join(page.get_text() for page in doc)
            if text.strip():
                return self.text_processor.clean_text(text)
        except Exception as e:
            logger.warning(f"PyMuPDF failed for {pdf_path}: {e}. Falling back to PyPDF2.")

        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            return self.text_processor.clean_text(text)
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path} with all methods: {e}")
            return ""

    def process_directory(self, pdf_directory: str) -> List[Document]:
        """Process all PDFs in a directory."""
        documents = []
        pdf_files = list(Path(pdf_directory).rglob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in '{pdf_directory}'.")

        for pdf_path in pdf_files:
            logger.info(f"Processing PDF: {pdf_path.name}")
            text = self.extract_text_from_pdf(str(pdf_path))
            if not text:
                continue

            chunks = self.text_processor.chunk_text(
                text, self.config.chunk_size, self.config.chunk_overlap_sentences
            )
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:
                    documents.append(Document(
                        content=chunk,
                        source=str(pdf_path),
                        doc_type='pdf',
                        chunk_id=f"pdf_{pdf_path.stem}_{i}",
                        metadata={'file_name': pdf_path.name}
                    ))
        logger.info(f"Extracted {len(documents)} total chunks from PDF files.")
        return documents

class JSONProcessor:
    """Handles JSON/JSONL data processing."""
    def __init__(self, text_processor: TurkishTextProcessor, config: RAGConfig):
        self.text_processor = text_processor
        self.config = config

    def process_file(self, json_path: str) -> List[Document]:
        """Process a single JSON or JSONL file."""
        documents = []
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f] if json_path.endswith('.jsonl') else json.load(f)
            if not isinstance(data, list): data = [data]
        except Exception as e:
            logger.error(f"Failed to load or parse JSON file {json_path}: {e}")
            return []

        for i, item in enumerate(data):
            content = ""
            if 'soru' in item and 'cevap' in item:
                content = f"Soru: {item['soru']}\nCevap: {item['cevap']}"
            elif 'text' in item:
                content = item['text']
            elif 'content' in item:
                content = item['content']
            else:
                content = ' '.join(str(v) for v in item.values() if isinstance(v, str))

            if content and len(content.strip()) > 20:
                doc = Document(
                    content=self.text_processor.clean_text(content),
                    source=json_path,
                    doc_type='json',
                    chunk_id=f"json_{Path(json_path).stem}_{i}",
                    metadata={'original_index': i}
                )
                documents.append(doc)
        return documents

    def process_directory(self, json_directory: str) -> List[Document]:
        """Process all JSON/JSONL files in a directory."""
        all_docs = []
        json_files = list(Path(json_directory).rglob("*.json")) + list(Path(json_directory).rglob("*.jsonl"))
        logger.info(f"Found {len(json_files)} JSON/JSONL files in '{json_directory}'.")
        for json_path in json_files:
            logger.info(f"Processing JSON: {json_path.name}")
            all_docs.extend(self.process_file(str(json_path)))
        logger.info(f"Extracted {len(all_docs)} total documents from JSON files.")
        return all_docs

class VectorStore:
    """Manages document embeddings and FAISS-based similarity search."""
    def __init__(self, config: RAGConfig, text_processor: TurkishTextProcessor):
        self.config = config
        self.text_processor = text_processor
        self.model = SentenceTransformer(config.model_name)
        self.documents: List[Document] = []
        self.index: Optional[faiss.Index] = None

    def build(self, documents: List[Document]):
        """Build the vector store from a list of documents."""
        if not documents:
            logger.warning("No documents provided to build vector store.")
            return

        self.documents = documents
        logger.info(f"Encoding {len(self.documents)} documents for the vector store...")

        texts_to_embed = [self.text_processor.preprocess_for_embedding(doc.content) for doc in self.documents]
        embeddings = self.model.encode(texts_to_embed, show_progress_bar=True, normalize_embeddings=True)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors.")

    def search(self, query: str) -> List[Tuple[Document, float]]:
        """Search for similar documents in the vector store."""
        if self.index is None or not self.documents:
            logger.warning("Search attempted but vector store is not built or is empty.")
            return []

        processed_query = self.text_processor.preprocess_for_embedding(query)
        query_embedding = self.model.encode([processed_query], normalize_embeddings=True)

        scores, indices = self.index.search(query_embedding.astype('float32'), self.config.retrieval_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= self.config.retrieval_score_threshold:
                results.append((self.documents[idx], float(score)))

        return results

    def save(self):
        """Save the vector store (index and documents) to disk."""
        path = Path(self.config.vector_store_path)
        path.mkdir(parents=True, exist_ok=True)

        if self.index:
            faiss.write_index(self.index, str(path / 'faiss_index.bin'))
        with open(path / 'documents.pkl', 'wb') as f:
            pickle.dump(self.documents, f)
        logger.info(f"Vector store saved to {self.config.vector_store_path}")

    def load(self) -> bool:
        """Load the vector store from disk."""
        path = Path(self.config.vector_store_path)
        index_path = path / 'faiss_index.bin'
        docs_path = path / 'documents.pkl'

        if not (index_path.exists() and docs_path.exists()):
            logger.info("No pre-existing vector store found to load.")
            return False

        self.index = faiss.read_index(str(index_path))
        with open(docs_path, 'rb') as f:
            self.documents = pickle.load(f)

        logger.info(f"Loaded vector store with {len(self.documents)} documents from {path}")
        return True

class FitnessRAG:
    """Main RAG system orchestrating processing, storage, and retrieval."""
    def __init__(self, config: RAGConfig = RAGConfig()):
        self.config = config
        self.text_processor = TurkishTextProcessor()
        self.pdf_processor = PDFProcessor(self.text_processor, self.config)
        self.json_processor = JSONProcessor(self.text_processor, self.config)
        self.vector_store = VectorStore(self.config, self.text_processor)

        if not self.vector_store.load():
            logger.info("No existing knowledge base found. Please build it using `build_knowledge_base`.")

    def build_knowledge_base(self, pdf_directory: str = None, json_directory: str = None):
        """Build the knowledge base from PDF and/or JSON directories."""
        all_documents = []
        if pdf_directory:
            all_documents.extend(self.pdf_processor.process_directory(pdf_directory))
        if json_directory:
            all_documents.extend(self.json_processor.process_directory(json_directory))

        if not all_documents:
            logger.warning("No documents were found in the specified directories. The knowledge base is empty.")
            return

        self.vector_store.build(all_documents)
        self.vector_store.save()

    def retrieve_context(self, query: str) -> str:
        """Retrieve and format context for a given query."""
        results = self.vector_store.search(query)
        if not results:
            return ""

        context_parts = []
        current_length = 0

        for doc, score in results:
            content = f"[Kaynak: {Path(doc.source).name}, Skor: {score:.2f}] {doc.content}"
            if current_length + len(content) > self.config.max_context_length:
                break
            context_parts.append(content)
            current_length += len(content)

        return "\n\n---\n\n".join(context_parts)

    def generate_rag_prompt(self, user_query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        """Generate an enhanced prompt with retrieved context."""
        context = self.retrieve_context(user_query)

        if context:
            prompt = f"""{system_prompt}

### BAĞLAMSAL BİLGİ KAYNAKLARI
{context}

### KULLANICI SORUSU
"{user_query}"

### CEVAP
"""
        else:
            prompt = f"""{system_prompt}

### KULLANICI SORUSU
"{user_query}"

### CEVAP
"""
        return prompt

    def get_statistics(self) -> Dict:
        """Get statistics about the knowledge base."""
        if not self.vector_store.documents:
            return {"status": "Knowledge base is not built."}

        total_docs = len(self.vector_store.documents)
        doc_types = [doc.doc_type for doc in self.vector_store.documents]
        sources = {Path(doc.source).name for doc in self.vector_store.documents}

        return {
            'total_chunks': total_docs,
            'pdf_chunks': doc_types.count('pdf'),
            'json_chunks': doc_types.count('json'),
            'unique_source_files': len(sources),
            'sources': sorted(list(sources))
        }

    def interactive_test(self):
        """Interactive testing interface."""
        print("\n" + "="*60)
        print("🏋️  FitTürkAI RAG Sistemi - İnteraktif Test Modu")
        print("="*60)

        stats = self.get_statistics()
        if 'status' in stats:
            print(f"❌ {stats['status']}")
            return

        print(f"📊 Bilgi Tabanı İstatistikleri:")
        print(f"   • Toplam chunk: {stats['total_chunks']}")
        print(f"   • PDF chunks: {stats['pdf_chunks']}")
        print(f"   • JSON chunks: {stats['json_chunks']}")
        print(f"   • Kaynak dosya sayısı: {stats['unique_source_files']}")
        print("\n" + "-"*60)
        print("💡 Test soruları yazın (çıkmak için 'quit' yazın)")
        print("-"*60)

        while True:
            try:
                user_query = input("\n🤔 Sorunuz: ").strip()

                if user_query.lower() in ['quit', 'exit', 'çık', 'q']:
                    print("👋 Görüşmek üzere!")
                    break

                if not user_query:
                    continue

                print("\n⏳ Aramalar yapılıyor...")
                start_time = time.time()

                # Retrieve context
                context = self.retrieve_context(user_query)
                search_time = time.time() - start_time

                print(f"⚡ Arama süresi: {search_time:.2f} saniye")

                if context:
                    print(f"\n📋 Bulunan Bağlam ({len(context)} karakter):")
                    print("-" * 40)
                    print(context[:500] + "..." if len(context) > 500 else context)
                    print("-" * 40)

                    print(f"\n🤖 Gelişmiş Prompt Oluşturuluyor...")
                    enhanced_prompt = self.generate_rag_prompt(user_query)
                    print(f"📝 Prompt uzunluğu: {len(enhanced_prompt)} karakter")
                else:
                    print("❌ Bu soru için kaynaklarda ilgili bilgi bulunamadı.")

            except KeyboardInterrupt:
                print("\n\n👋 Program sonlandırıldı!")
                break
            except Exception as e:
                print(f"❌ Hata oluştu: {e}")


def main():
    """Ana fonksiyon - Gerçek veri ile test."""
    print("🚀 FitTürkAI RAG Sistemi - Gerçek Veri Testi")
    print("="*50)

    # Force download NLTK data at startup
    print("📦 NLTK verilerini indiriliyor...")
    try:
        import nltk
        nltk.download('punkt_tab', quiet=True)
        print("✅ punkt_tab indirildi")
    except:
        try:
            nltk.download('punkt', quiet=True)
            print("✅ punkt indirildi")
        except:
            print("⚠️  NLTK tokenizer indirilemedi, temel metin bölme kullanılacak")

    try:
        nltk.download('stopwords', quiet=True)
        print("✅ stopwords indirildi")
    except:
        print("⚠️  NLTK stopwords indirilemedi, temel liste kullanılacak")

    # Konfigürasyon
    config = RAGConfig(
        vector_store_path="./real_fitness_rag_store",
        retrieval_k=5,
        retrieval_score_threshold=0.15,  # Biraz daha düşük threshold
        max_context_length=3000
    )

    # RAG sistemini başlat
    rag = FitnessRAG(config)

    # Gerçek veri yolları
    JSON_PATH = "./DATA"  # train.json dosyasının bulunduğu klasör
    PDF_PATH = "./indirilen_pdfler"  # PDF'lerin bulunduğu klasör

    # Dosya varlığını kontrol et
    json_exists = Path(JSON_PATH).exists()
    pdf_exists = Path(PDF_PATH).exists()

    print(f"📁 JSON klasörü ({JSON_PATH}): {'✅ Var' if json_exists else '❌ Yok'}")
    print(f"📁 PDF klasörü ({PDF_PATH}): {'✅ Var' if pdf_exists else '❌ Yok'}")

    if not json_exists and not pdf_exists:
        print("❌ Hiçbir veri klasörü bulunamadı! Lütfen yolları kontrol edin.")
        return

    # Mevcut bilgi tabanını kontrol et
    if Path(config.vector_store_path).exists():
        print(f"\n💾 Mevcut bilgi tabanı bulundu: {config.vector_store_path}")
        choice = input("Yeniden oluşturulsun mu? (y/N): ").strip().lower()
        if choice == 'y':
            print("🔄 Bilgi tabanı yeniden oluşturuluyor...")
            import shutil
            shutil.rmtree(config.vector_store_path)
            start_time = time.time()
            rag.build_knowledge_base(
                pdf_directory=PDF_PATH if pdf_exists else None,
                json_directory=JSON_PATH if json_exists else None
            )
            build_time = time.time() - start_time
            print(f"✅ Bilgi tabanı oluşturuldu! ({build_time:.2f} saniye)")
    else:
        print("\n🔨 Bilgi tabanı oluşturuluyor...")
        start_time = time.time()
        rag.build_knowledge_base(
            pdf_directory=PDF_PATH if pdf_exists else None,
            json_directory=JSON_PATH if json_exists else None
        )
        build_time = time.time() - start_time
        print(f"✅ Bilgi tabanı oluşturuldu! ({build_time:.2f} saniye)")

    # İstatistikleri göster
    stats = rag.get_statistics()
    print(f"\n📊 Final İstatistikler:")
    for key, value in stats.items():
        if key == 'sources':
            print(f"   • {key}: {len(value)} dosya")
            if len(value) <= 10:
                for source in value:
                    print(f"     - {source}")
            else:
                for source in value[:5]:
                    print(f"     - {source}")
                print(f"     ... ve {len(value)-5} dosya daha")
        else:
            print(f"   • {key}: {value}")

    # Test sorularını çalıştır
    test_queries = [
        "Protein tozu nasıl kullanılır?",
        "Kilo vermek için hangi egzersizler yapmalıyım?",
        "Kas geliştirmek için ne kadar protein tüketmeliyim?",
        "Antrenman öncesi ne yemeli?",
        "Kardiyo egzersizlerin faydaları nelerdir?",
        "Yaralanmadan nasıl korunurum?",
        "Beslenme programı nasıl hazırlanır?",
        "Hangi vitaminleri almalıyım?"
    ]

    print(f"\n🧪 Örnek Test Sorularını Çalıştırıyor...")
    print("-" * 50)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. SORU: {query}")
        start_time = time.time()
        context = rag.retrieve_context(query)
        search_time = time.time() - start_time

        if context:
            print(f"   ✅ Bağlam bulundu ({len(context)} karakter) - {search_time:.3f}s")
            # İlk 150 karakteri göster
            preview = context.replace('\n', ' ')[:150] + "..." if len(context) > 150 else context
            print(f"   📝 Önizleme: {preview}")
        else:
            print(f"   ❌ Bağlam bulunamadı - {search_time:.3f}s")

    # İnteraktif test modu
    print(f"\n🎯 İnteraktif test moduna geçiliyor...")
    rag.interactive_test()


if __name__ == "__main__":
    main()
