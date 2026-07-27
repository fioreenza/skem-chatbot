import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

# Memuat API Key dari file .env
load_dotenv()

# Memastikan API Key tersedia sebelum menjalankan program
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY tidak ditemukan di file .env. Harap isi terlebih dahulu.")

# Impor dari pustaka yang diperlukan
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document  
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Membuat kelas untuk menyimpan dan mencocokkan pertanyaan yang mirip secara semantik
class LocalSemanticCache:
    """
    Menyimpan dan mencocokkan pertanyaan yang mirip secara semantik 
    menggunakan FAISS untuk menghemat biaya token OpenAI.
    """
    def __init__(self, embeddings, threshold=0.15, cache_dir="../semantic_cache_db"):
        self.embeddings = embeddings
        self.threshold = threshold  
        self.cache_dir = cache_dir
        self.vector_store = None
        self.load_cache()

    def load_cache(self):
        """Memuat database cache jika sudah ada di lokal."""
        if os.path.exists(self.cache_dir):
            try:
                self.vector_store = FAISS.load_local(
                    self.cache_dir, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                print("[CACHE-INFO] Berhasil memuat semantic cache dari lokal.")
            except Exception as e:
                print(f"[CACHE-ERROR] Gagal memuat cache: {e}. Membuat cache baru.")
                self.vector_store = None

    def save_cache(self):
        """Menyimpan database cache ke penyimpanan lokal."""
        if self.vector_store:
            try:
                self.vector_store.save_local(self.cache_dir)
                print("[CACHE-INFO] Berhasil menyimpan semantic cache ke lokal.")
            except Exception as e:
                print(f"[CACHE-ERROR] Gagal menyimpan cache ke disk: {e}")

    def lookup(self, prompt: str):
        """Mencari apakah ada pertanyaan serupa yang sudah pernah dijawab sebelumnya."""
        if not self.vector_store:
            return None
        try:
            # Menggunakan k=1 untuk mengambil 1 pertanyaan paling mirip
            results = self.vector_store.similarity_search_with_score(prompt, k=1)
            if results:
                doc, score = results[0]
                # FAISS L2 Distance: Semakin mendekati 0 berarti semakin identik pertanyaannya.
                if score <= self.threshold:
                    return {
                        "jawaban": doc.metadata["jawaban"],
                        "sumber": doc.metadata["sumber"],
                        "score": score
                    }
        except Exception as e:
            print(f"[CACHE-ERROR] Terjadi kesalahan saat lookup cache: {e}")
        return None

    def add(self, prompt: str, jawaban: str, sumber: list):
        """Menyimpan pertanyaan dan jawaban baru ke dalam database cache."""
        doc = Document(
            page_content=prompt,
            metadata={"jawaban": jawaban, "sumber": sumber}
        )
        try:
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents([doc], self.embeddings)
            else:
                self.vector_store.add_documents([doc])
            self.save_cache()
        except Exception as e:
            print(f"[CACHE-ERROR] Gagal menyimpan ke cache: {e}")


# Variabel global untuk menyimpan model RAG dan Semantic Cache setelah dimuat
rag_pipeline = None
semantic_cache = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Fungsi siklus hidup FastAPI untuk menyiapkan database vektor dan semantic cache
    sekali saja saat web server pertama kali dinyalakan.
    """
    global rag_pipeline, semantic_cache
    
    if not os.path.exists("data"):
        raise FileNotFoundError("Folder 'data' tidak ditemukan. Harap buat folder 'data' dan letakkan file PDF di dalamnya.")

    print(f"\n--- [Mulai] Menginisialisasi RAG untuk file PDF di folder 'data' ---\n")
    try:
        # Memuat dokumen pdf
        loader = DirectoryLoader(
            "data",
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        
        # Memotong teks (chunking) agar lebih mudah diproses oleh LLM   
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,       
            chunk_overlap=200,     
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        # Membuat embeddings dan menyimpan ke FAISS vector store
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Inisialisasi semantic cache lokal untuk menyimpan pertanyaan yang mirip secara semantik
        semantic_cache = LocalSemanticCache(embeddings=embeddings, threshold=0.15)

        # Membuat retriever dari vector store untuk digunakan dalam pipeline RAG
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Membuat LLM ChatOpenAI dengan model GPT-4o-mini 
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        system_prompt = (
            "Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan dokumen yang diberikan saja.\n"
            "Gunakan potongan konteks berikut untuk menjawab pertanyaan.\n"
            "Jika Anda tidak mengetahui jawabannya atau jawabannya tidak terdapat dalam dokumen, "
            "maka Anda WAJIB menjawab: 'Maaf, saya tidak tahu karena informasi tersebut tidak ada di dalam dokumen.'\n"
            "Jangan pernah mengarang jawaban sendiri di luar konteks yang disediakan.\n\n"
            "Jawab walaupun bahasa yang digunakan dalam pertanyaan bukan bahasa Indonesia, tetapi tetap gunakan bahasa yang sama dengan pertanyaan.\n\n"
            "Konteks:\n"
            "{context}"
        )
        
        # Memperbarui template prompt agar mendukung riwayat percakapan (chat_history)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_pipeline = create_retrieval_chain(retriever, question_answer_chain)
        print("--- [Selesai] Alur RAG Berhasil Dikonfigurasi & Siap Digunakan! ---\n")
    except Exception as e:
        print(f"[Error] Gagal melakukan inisialisasi RAG: {e}\n")
        
    yield

# Inisialisasi aplikasi FastAPI dengan siklus hidup di atas
app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)

# CORS Middleware agar frontend website Anda nanti bebas melakukan request ke backend ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Menentukan bentuk data JSON yang diterima dari website
class TanyaRequest(BaseModel):
    pertanyaan: str
    riwayat: List[Dict[str, str]] = []

@app.post("/tanya")
async def tanya_chatbot(request: TanyaRequest):
    global rag_pipeline, semantic_cache
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="Sistem RAG belum siap atau gagal dimuat.")
    
    if not request.pertanyaan.strip():
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong.")
        
    try:
        # Log terminal untuk menampilkan pertanyaan dan riwayat chat yang diterima
        print("\n" + "="*60)
        print("[DATABASE LOG] Permintaan Pertanyaan Diterima:")
        print(f"  - Pertanyaan Saat Ini : '{request.pertanyaan}'")
        print(f"  - Total Riwayat Chat  : {len(request.riwayat)} pesan")
        if request.riwayat:
            print("  - Detail Isi Riwayat  :")
            for index, msg in enumerate(request.riwayat):
                role_label = msg.get('role', 'unknown').upper()
                content_preview = msg.get('content', '')[:60].replace('\n', ' ')
                print(f"    {index + 1}. [{role_label}]: \"{content_preview}...\"")
        print("="*60)

        # Periksa semantic cache untuk pertanyaan yang mirip agar menghemat biaya token OpenAI
        if semantic_cache:
            cache_hit = semantic_cache.lookup(request.pertanyaan)
            if cache_hit:
                # Menampilkan log saat cache berhasil ditemukan
                print("\n" + "⚡"*30)
                print(f"[CACHE HIT] Ditemukan kecocokan semantik (Skor Jarak: {cache_hit['score']:.4f})")
                print(f"  - Mengembalikan jawaban langsung dari cache lokal tanpa biaya API.")
                print("⚡"*30 + "\n")
                
                return {
                    "status": "success",
                    "jawaban": cache_hit["jawaban"],
                    "sumber": cache_hit["sumber"],
                    "cached": True
                }

        # Mengubah format riwayat dari JSON [{"role": "user", "content": "..."}] 
        # menjadi objek pesan resmi LangChain agar dipahami oleh LLM
        chat_history = []
        for msg in request.riwayat:
            if msg.get("role") == "user":
                chat_history.append(HumanMessage(content=msg.get("content")))
            elif msg.get("role") == "assistant" or msg.get("role") == "ai":
                chat_history.append(AIMessage(content=msg.get("content")))

        # Memanggil pipeline RAG dengan input pertanyaan & chat_history
        response = rag_pipeline.invoke({
            "input": request.pertanyaan,
            "chat_history": chat_history
        })
        
        # Menyusun data rujukan
        sumber_rujukan = []
        for doc in response.get("context", []):
            path_sumber = doc.metadata.get('source', 'Tidak diketahui')
            nama_dokumen = os.path.basename(path_sumber)
            page_num = doc.metadata.get('page', 0) + 1
            kutipan = doc.page_content.strip().replace("\n", " ")
            if len(kutipan) > 150:
                kutipan = kutipan[:150] + "..."
            
            sumber_rujukan.append({
                "dokumen": nama_dokumen,       # Nama file PDF
                "path_lengkap": path_sumber,   # Path lengkap (misal: data/laporan.pdf)
                "halaman": page_num,
                "kutipan": kutipan
            })

        # Simpan jawaban baru ke semantic cache
        if semantic_cache:
            # Hindari menyimpan jawaban fallback error/tidak tahu ke dalam cache
            fallback_msg = "Maaf, saya tidak tahu karena informasi tersebut tidak ada di dalam dokumen."
            if fallback_msg not in response["answer"]:
                semantic_cache.add(request.pertanyaan, response["answer"], sumber_rujukan)
                print(f"[CACHE UPDATE] Pertanyaan baru berhasil didaftarkan ke database cache.")

        # Log terminal untuk menampilkan jawaban dan rujukan yang dihasilkan
        print("\n" + "="*60)
        print("[DATABASE LOG] Hasil Pemrosesan LLM & RAG (CACHE MISS):")
        print(f"  - Jawaban Dihasilkan  : '{response['answer']}'")
        print(f"  - Dokumen Ditemukan   : {len(sumber_rujukan)} rujukan")
        for index, ref in enumerate(sumber_rujukan):
            print(f"    * {ref['dokumen']} (Hal. {ref['halaman']}) -> \"{ref['kutipan'][:60]}...\"")
        print("="*60 + "\n")
            
        return {
            "status": "success",
            "jawaban": response["answer"],
            "sumber": sumber_rujukan,
            "cached": False
        }
    except Exception as e:
        # Log terminal untuk menampilkan bila terjadi kesalahan
        print("\n" + "!"*60)
        print(f"[ERROR LOG] Kesalahan fatal saat memproses: {str(e)}")
        print("!"*60 + "\n")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses: {str(e)}")