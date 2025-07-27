# rag_implementation.py
# Extracted and cleaned from Code.ipynb

import PyPDF2
import pdfplumber
import pandas as pd
import numpy as np
import re
import warnings
from typing import List, Dict, Any
import time
import json
from collections import defaultdict, Counter

# Install missing packages automatically
def install_package(package):
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except:
        pass

# Try to import required packages, install if missing
try:
    import torch
except ImportError:
    install_package("torch")
    import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    install_package("sentence-transformers")
    from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    install_package("faiss-cpu")
    import faiss

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    install_package("scikit-learn")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    install_package("rank_bm25")
    from rank_bm25 import BM25Okapi

warnings.filterwarnings('ignore')

class PDFProcessor:
    def __init__(self):
        pass

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text

    def clean_text(self, text):
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\$\%\.\,\-\(\)]', ' ', text)
        text = text.replace('$', '$ ')
        text = text.replace('%', ' %')
        return text.strip()

    def chunk_text(self, text, chunk_size=500, overlap=50):
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 0:
                chunks.append(chunk.strip())

        return chunks

class VectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.chunks = None
        self.index = None

    def create_embeddings(self, chunks):
        print(f"Creating embeddings for {len(chunks)} chunks...")
        self.chunks = chunks
        self.embeddings = self.model.encode(chunks, show_progress_bar=False)

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        faiss.normalize_L2(self.embeddings.astype('float32'))
        self.index.add(self.embeddings.astype('float32'))

        print(f"Vector store created with {len(chunks)} chunks")

    def search(self, query, top_k=3):
        if self.index is None:
            raise ValueError("Vector store not initialized. Call create_embeddings first.")

        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding.astype('float32'))

        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            results.append({
                'chunk': self.chunks[idx],
                'score': float(score),
                'rank': i + 1
            })

        return results

class BasicGenerator:
    def __init__(self):
        pass

    def generate_answer(self, query, context_chunks, max_length=150):
        context = "\n".join([chunk['chunk'] for chunk in context_chunks[:3]])
        query_lower = query.lower()

        if 'revenue' in query_lower:
            revenue_match = re.search(r'\$\s*(\d+[\d,]*)\s*(?:million|billion)', context, re.IGNORECASE)
            if revenue_match:
                return f"Based on the financial report, the revenue was ${revenue_match.group(1)} million."

        if 'income' in query_lower:
            income_match = re.search(r'(?:net income|income).*?\$\s*(\d+[\d,]*)', context, re.IGNORECASE)
            if income_match:
                return f"According to the report, the net income was ${income_match.group(1)} million."

        if 'financial highlights' in query_lower or 'key highlights' in query_lower:
            highlights = []

            revenue_match = re.search(r'Revenue.*?\$\s*(\d+[\d,]*)', context, re.IGNORECASE)
            if revenue_match:
                highlights.append(f"Revenue: ${revenue_match.group(1)} million")

            income_match = re.search(r'Net income.*?\$\s*(\d+[\d,]*)', context, re.IGNORECASE)
            if income_match:
                highlights.append(f"Net income: ${income_match.group(1)} million")

            growth_match = re.search(r'(\d+)\s*%.*?(?:increase|growth)', context, re.IGNORECASE)
            if growth_match:
                highlights.append(f"Growth: {growth_match.group(1)}%")

            if highlights:
                return "Key financial highlights: " + "; ".join(highlights)

        if context_chunks:
            best_chunk = context_chunks[0]['chunk']
            return f"Based on the document: {best_chunk[:200]}..."

        return "I couldn't find specific information to answer your question."

class BasicRAGPipeline:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.generator = BasicGenerator()

    def setup(self, pdf_path):
        print("Setting up Basic RAG pipeline...")

        raw_text = self.pdf_processor.extract_text_from_pdf(pdf_path)
        clean_text = self.pdf_processor.clean_text(raw_text)
        chunks = self.pdf_processor.chunk_text(clean_text)

        print(f"Processed {len(chunks)} text chunks")

        self.vector_store.create_embeddings(chunks)

        print("Basic RAG pipeline setup complete!")

    def query(self, question, top_k=3):
        start_time = time.time()
        print(f"Query: {question}")

        retrieved_chunks = self.vector_store.search(question, top_k)

        print(f"Retrieved {len(retrieved_chunks)} relevant chunks")

        answer = self.generator.generate_answer(question, retrieved_chunks)

        query_time = time.time() - start_time
        print(f"Answer: {answer}")
        return {
            'question': question,
            'answer': answer,
            'retrieved_chunks': retrieved_chunks,
            'search_results': retrieved_chunks,
            'query_time': query_time
        }

# Enhanced classes with simplified implementations
class EnhancedPDFProcessor(PDFProcessor):
    def __init__(self):
        super().__init__()
        self.text_chunks = []
        self.tables = []
        self.structured_data = {}

    def extract_text_and_tables(self, pdf_path):
        print("Extracting text and tables from PDF...")
        text = self.extract_text_from_pdf(pdf_path)
        tables = self._extract_tables_pdfplumber(pdf_path)
        return text, tables

    def _extract_tables_pdfplumber(self, pdf_path):
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table_num, table in enumerate(page_tables):
                        if table and len(table) > 1:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            df = self._clean_table_dataframe(df)
                            table_info = {
                                'page': page_num + 1,
                                'table_id': f"page_{page_num+1}_table_{table_num+1}",
                                'dataframe': df,
                                'raw_data': table
                            }
                            tables.append(table_info)
        except Exception as e:
            print(f"Error extracting tables: {e}")
        
        print(f"Extracted {len(tables)} tables")
        return tables

    def _clean_table_dataframe(self, df):
        df = df.dropna(how='all').dropna(axis=1, how='all')
        df.columns = [str(col).strip().replace('\n', ' ') if col else f"Column_{i}"
                     for i, col in enumerate(df.columns)]
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip().str.replace('\n', ' ')
        
        return df

    def process_structured_data(self, tables):
        structured_data = {}
        for table_info in tables:
            table_id = table_info['table_id']
            df = table_info['dataframe']
            structured_data[table_id] = {
                'dataframe': df,
                'page': table_info['page'],
                'text_representation': self._table_to_text(df),
                'key_value_pairs': {},
                'financial_metrics': {}
            }
        return structured_data

    def _table_to_text(self, df):
        text_parts = ["Table with columns: " + ", ".join(df.columns)]
        for _, row in df.iterrows():
            row_text = []
            for col, value in row.items():
                if pd.notna(value) and str(value).strip():
                    row_text.append(f"{col}: {value}")
            if row_text:
                text_parts.append(" | ".join(row_text))
        return "\n".join(text_parts)

class EnhancedRAGPipeline:
    def __init__(self):
        self.pdf_processor = EnhancedPDFProcessor()
        self.vector_store = VectorStore()
        self.generator = BasicGenerator()
        self.text_chunks = []
        self.structured_data = {}

    def setup(self, pdf_path):
        print("Setting up Enhanced RAG Pipeline")
        
        try:
            # Extract text and tables
            raw_text, tables = self.pdf_processor.extract_text_and_tables(pdf_path)
            
            # Process text chunks
            clean_text = self.pdf_processor.clean_text(raw_text)
            self.text_chunks = self.pdf_processor.chunk_text(clean_text)
            
            # Process structured data
            self.structured_data = self.pdf_processor.process_structured_data(tables)
            
            # Create embeddings
            self.vector_store.create_embeddings(self.text_chunks)
            
            print(f"Enhanced RAG Pipeline setup complete!")
            print(f"   - Text chunks: {len(self.text_chunks)}")
            print(f"   - Tables: {len(self.structured_data)}")
            
        except Exception as e:
            print(f"Error during enhanced setup: {e}")
            raise e

    def query(self, question, top_k=3):
        start_time = time.time()
        print(f"Enhanced Query: {question}")
        
        # Get search results
        search_results = self.vector_store.search(question, top_k)
        
        # Generate answer
        answer = self.generator.generate_answer(question, search_results)
        
        query_time = time.time() - start_time
        
        print(f"Enhanced Answer: {answer}")
        print(f"Total time: {query_time:.3f}s")
        
        return {
            'question': question,
            'answer': answer,
            'search_results': search_results,
            'query_time': query_time,
            'performance_metrics': {
                'num_text_results': len(search_results),
                'num_structured_results': len(self.structured_data),
                'avg_relevance_score': sum(r.get('score', 0) for r in search_results) / len(search_results) if search_results else 0
            }
        }

# Advanced RAG Pipeline with query optimization
class AdvancedRAGPipeline:
    def __init__(self):
        self.pdf_processor = EnhancedPDFProcessor()
        self.vector_store = VectorStore()
        self.generator = BasicGenerator()
        self.text_chunks = []
        self.structured_data = {}
        self.setup_complete = False

    def setup(self, pdf_path, chunk_size=500, overlap=50):
        print("Setting up Advanced RAG Pipeline")
        
        try:
            # Extract text and tables
            raw_text, tables = self.pdf_processor.extract_text_and_tables(pdf_path)
            
            # Process text chunks
            clean_text = self.pdf_processor.clean_text(raw_text)
            self.text_chunks = self.pdf_processor.chunk_text(clean_text, chunk_size, overlap)
            
            # Process structured data
            self.structured_data = self.pdf_processor.process_structured_data(tables)
            
            # Create embeddings
            self.vector_store.create_embeddings(self.text_chunks)
            
            self.setup_complete = True
            
            print(f"Advanced RAG Pipeline setup complete!")
            print(f"   - Text chunks: {len(self.text_chunks)}")
            print(f"   - Tables: {len(self.structured_data)}")
            
        except Exception as e:
            print(f"Error during advanced setup: {e}")
            raise e

    def _optimize_query(self, question):
        """Simple query optimization"""
        # Add financial synonyms
        synonyms = {
            'revenue': ['sales', 'income', 'earnings'],
            'profit': ['income', 'earnings', 'gains'],
            'expenses': ['costs', 'expenditures', 'spending']
        }
        
        optimized = question.lower()
        for term, syns in synonyms.items():
            if term in optimized:
                optimized += " " + " ".join(syns)
        
        return optimized

    def query(self, question, top_k=5):
        if not self.setup_complete:
            raise ValueError("Pipeline not setup. Call setup() first.")
        
        start_time = time.time()
        print(f"Advanced Query: {question}")
        
        # Optimize query
        optimized_question = self._optimize_query(question)
        
        # Get search results
        search_results = self.vector_store.search(optimized_question, top_k)
        
        # Enhanced answer generation
        answer = self._generate_enhanced_answer(question, search_results)
        
        query_time = time.time() - start_time
        
        print(f"Advanced Answer: {answer}")
        print(f"Total time: {query_time:.3f}s")
        
        return {
            'question': question,
            'answer': answer,
            'search_results': search_results,
            'query_time': query_time,
            'optimized_query': {'query_type': self._classify_query(question)},
            'performance_metrics': {
                'num_text_results': len(search_results),
                'num_structured_results': len(self.structured_data),
                'avg_relevance_score': sum(r.get('score', 0) for r in search_results) / len(search_results) if search_results else 0
            }
        }

    def _classify_query(self, query):
        """Classify query type"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['compared to', 'vs', 'versus', 'difference', 'change']):
            return 'comparative'
        elif any(word in query_lower for word in ['summarize', 'summary', 'overview', 'breakdown']):
            return 'summary'
        elif any(word in query_lower for word in ['revenue', 'income', 'expenses', 'margin', 'profit']):
            return 'financial_metric'
        else:
            return 'general'

    def _generate_enhanced_answer(self, query, search_results):
        """Generate enhanced answers based on query type"""
        query_type = self._classify_query(query)
        
        if query_type == 'comparative':
            return self._generate_comparative_answer(query, search_results)
        elif query_type == 'summary':
            return self._generate_summary_answer(query, search_results)
        else:
            return self.generator.generate_answer(query, search_results)

    def _generate_comparative_answer(self, query, search_results):
        """Generate comparative answers"""
        context = "\n".join([chunk['chunk'] for chunk in search_results[:3]])
        
        # Look for year-over-year comparisons
        if 'income' in query.lower() and ('2024' in query or '2023' in query):
            income_2024 = re.search(r'(?:2024).*?(?:income|net).*?\$?\s*([0-9,]+)', context, re.IGNORECASE)
            income_2023 = re.search(r'(?:2023).*?(?:income|net).*?\$?\s*([0-9,]+)', context, re.IGNORECASE)
            
            if income_2024 and income_2023:
                return f"Based on the financial data, net income was ${income_2024.group(1)} million in 2024 compared to ${income_2023.group(1)} million in 2023."
        
        return f"Based on the financial data: {context[:200]}..."

    def _generate_summary_answer(self, query, search_results):
        """Generate summary answers"""
        context = "\n".join([chunk['chunk'] for chunk in search_results[:3]])
        
        if 'expenses' in query.lower():
            # Look for expense breakdown
            expenses = []
            expense_patterns = [
                r'(?:cost of revenue|cost).*?\$?\s*([0-9,]+)',
                r'(?:research|R&D).*?\$?\s*([0-9,]+)',
                r'(?:marketing|sales).*?\$?\s*([0-9,]+)',
                r'(?:administrative|admin).*?\$?\s*([0-9,]+)'
            ]
            
            for pattern in expense_patterns:
                match = re.search(pattern, context, re.IGNORECASE)
                if match:
                    expenses.append(f"${match.group(1)} million")
            
            if expenses:
                return f"Operating expenses breakdown: {'; '.join(expenses[:4])}"
        
        # Default summary
        sentences = context.split('.')[:3]
        return "Summary: " + ". ".join([s.strip() for s in sentences if len(s.strip()) > 20])

print("✅ RAG Implementation loaded successfully!")
print("Available classes: BasicRAGPipeline, EnhancedRAGPipeline, AdvancedRAGPipeline")