import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_knowledge_base(file_path: str):
    """
    Verilen dosya yolundan JSON verisini yukler.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Hata: '{file_path}' dosyasi bulunamadi.")
        return []
    except json.JSONDecodeError:
        print(f"Hata: '{file_path}' dosyasi gecerli bir JSON formatinda degil.")
        return []

class SimpleRAG:
    """
    Verilen bilgi bankasi ile RAG islemlerini yurutem basit bir sinif.
    """
    def __init__(self, knowledge_base: list):
        self.knowledge_base = knowledge_base
        self.documents = [item['cevap'] for item in self.knowledge_base]
        self.questions = [item['soru'] for item in self.knowledge_base]
        self.vectorizer = TfidfVectorizer(stop_words=None)
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)
        print("Bilgi bankasi basariyla vektorleştirildi.")

    def get_relevant_context(self, query: str, top_k: int = 1):
        """
        Verilen bir soruya en cok benzeyen cevabi (context'i) bulur.
        """
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        relevant_doc_indices = cosine_similarities.argsort()[-top_k:][::-1]
        relevant_docs = [self.documents[i] for i in relevant_doc_indices]
        return relevant_docs

    def create_prompt_for_llm(self, query: str, context: list):
        """
        LLM'e gonderilmek uzere, context ile zenginleştirilmiş bir prompt olusturur.
        """
        context_str = "\n\n".join(context)
        prompt = f"""
        Asagidaki bilgileri (CONTEXT) kullanarak kullanici sorusunu (QUESTION) yanitlayin.
        Cevabiniz yalnizca verilen CONTEXT'e dayanmalidir.

        CONTEXT:
        ---
        {context_str}
        ---

        QUESTION: {query}

        ANSWER:
        """
        return prompt

if __name__ == "__main__":
    file_name = 'Kisisellestirmis Beslenme ve Egzersiz Planlari.json'
    knowledge_base_data = load_knowledge_base(file_name)

    if knowledge_base_data:
        rag_system = SimpleRAG(knowledge_base_data)
        user_query = "Tip 2 diyabette alkol neden risklidir?"
        print(f"\nKullanici Sorusu: '{user_query}'")
        retrieved_context = rag_system.get_relevant_context(user_query)
        print("\n--- Adim 1: En Ilgili Bilgi Getirildi (Retrieval) ---")
        for i, doc in enumerate(retrieved_context):
            print(f"Ilgili Bilgi {i+1}: {doc}")
        print("------------------------------------------------------")
        final_prompt = rag_system.create_prompt_for_llm(user_query, retrieved_context)
        print("\n--- Adim 2: LLM Icin Hazirlanan Zenginleştirilmiş Prompt (Augmented Prompt) ---")
        print(final_prompt)
        print("----------------------------------------------------------------------------")
        print("\nBu son prompt, bir dil modeline (orn: Google Gemini) gönderilmeye hazırdır.")