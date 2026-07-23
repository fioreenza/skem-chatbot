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

# Impor dari pustaka yang sudah berhasil Anda gunakan sebelumnya
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Variabel global untuk menyimpan model RAG setelah dimuat
rag_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Fungsi siklus hidup FastAPI untuk menyiapkan database vektor sekali saja
    saat web server pertama kali dinyalakan.
    """
    global rag_pipeline
    # Baca smeua file PDF di folder data
    
    if not os.path.exists("data"):
        raise FileNotFoundError("Folder 'data' tidak ditemukan. Harap buat folder 'data' dan letakkan file PDF di dalamnya.")

    print(f"\n--- [Mulai] Menginisialisasi RAG untuk file PDF di folder 'data' ---\n")
    try:
        # 1. MEMBACA PDF
        loader = DirectoryLoader(
            "data",
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        
        # 2. MEMOTONG TEKS (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,       
            chunk_overlap=200,     
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        # 3. MEMBUAT EMBEDDING & VECTOR STORE
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)

        # 4. RETRIEVER
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # 5. PROMPTING & CHATBOT SETUP 
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
    global rag_pipeline
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="Sistem RAG belum siap atau gagal dimuat.")
    
    if not request.pertanyaan.strip():
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong.")
        
    try:
        # --- LOG TERMINAL: MENAMPILKAN INPUT YANG MASUK ---
        print("\n" + "="*60)
        print("[DATABASE LOG] Permintaan Pertanyaan Diterima:")
        print(f"  - Pertanyaan Saat Ini : '{request.pertanyaan}'")
        print(f"  - Total Riwayat Chat  : {len(request.riwayat)} pesan")
        if request.riwayat:
            print("  - Detail Isi Riwayat  :")
            for index, msg in enumerate(request.riwayat):
                # Cetak ringkasan isi chat lama
                role_label = msg.get('role', 'unknown').upper()
                content_preview = msg.get('content', '')[:60].replace('\n', ' ')
                print(f"    {index + 1}. [{role_label}]: \"{content_preview}...\"")
        print("="*60)

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
            # Mengambil path lengkap file dari metadata 'source'
            path_sumber = doc.metadata.get('source', 'Tidak diketahui')
            
            # Mengambil nama file saja (misal: "laporan.pdf") dari path lengkap
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

        # --- LOG TERMINAL: MENAMPILKAN HASIL PEMROSESAN ---
        print("\n" + "="*60)
        print("[DATABASE LOG] Hasil Pemrosesan LLM & RAG:")
        print(f"  - Jawaban Dihasilkan  : '{response['answer']}'")
        print(f"  - Dokumen Ditemukan   : {len(sumber_rujukan)} rujukan")
        for index, ref in enumerate(sumber_rujukan):
            # Sekarang log terminal menampilkan nama dokumen juga
            print(f"    * {ref['dokumen']} (Hal. {ref['halaman']}) -> \"{ref['kutipan'][:60]}...\"")
        print("="*60 + "\n")
            
        return {
            "status": "success",
            "jawaban": response["answer"],
            "sumber": sumber_rujukan
        }
    except Exception as e:
        # --- LOG TERMINAL: MENAMPILKAN BILA TERJADI KESALAHAN ---
        print("\n" + "!"*60)
        print(f"[ERROR LOG] Kesalahan fatal saat memproses: {str(e)}")
        print("!"*60 + "\n")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses: {str(e)}")

## to do
# Semantic Caching (Penyimpanan Riwayat Jawaban) Untuk menghemat biaya, developer chatbot menggunakan Semantic Caching (misalnya menggunakan Redis atau GPTCache).Cara kerja: Ketika pengguna bertanya, sistem akan memeriksa database cache terlebih dahulu. Jika ada pertanyaan yang sangat mirip yang pernah dijawab sebelumnya, sistem akan langsung memberikan jawaban tersebut tanpa mengirimkan request ke OpenAI. Ini bisa menghemat biaya token hingga 30% - 50%.