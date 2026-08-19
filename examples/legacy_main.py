import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
app = FastAPI(title="RAG作业接口")

UPLOAD_FOLDER = "./upload_files"
PERSIST_DIR = "./chroma_store"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 60
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

embedding = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="qwen2.5:3b", temperature=0)

vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    return {"code":0, "msg":"文件保存成功", "filename":file.filename, "save_path":save_path}

def load_txt_file(filepath:str):
    """兼容utf‑8 / gbk编码读取txt"""
    try:
        with open(filepath,"r",encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filepath,"r",encoding="gbk") as f:
            return f.read()

@app.post("/vectorize/single")
async def vector_single(filename:str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return {"code":-1,"msg":"文件不存在"}
    content = load_txt_file(file_path)
    splits = text_splitter.split_text(content)
    vector_db.add_texts(texts=splits, metadatas=[{"source":filename}]*len(splits))
    return {"code":0, "msg":"单文件向量化完成","chunk_count":len(splits)}

@app.post("/vectorize/batch")
async def vector_batch():
    all_splits = []
    all_meta = []
    for fname in os.listdir(UPLOAD_FOLDER):
        if fname.endswith((".txt",".md")):
            fp = os.path.join(UPLOAD_FOLDER,fname)
            content = load_txt_file(fp)
            chunks = text_splitter.split_text(content)
            all_splits.extend(chunks)
            all_meta.extend([{"source":fname}]*len(chunks))
    vector_db.add_texts(texts=all_splits, metadatas=all_meta)
    return {"code":0, "msg":"批量向量化完成","total_chunk":len(all_splits)}

prompt = ChatPromptTemplate.from_messages([
    ("system","基于下面上下文回答用户问题，如果上下文没有答案就如实说不知道。上下文：{context}"),
    ("human","用户问题：{question}")
])
rag_chain = prompt | llm | StrOutputParser()

@app.post("/rag/query")
async def rag_query(question:str):
    retriever = vector_db.as_retriever(search_kwargs={"k":3})
    docs = retriever.invoke(question)
    context_text = "\n".join([d.page_content for d in docs])
    # 调试打印，观察传给大模型的上下文
    print("====传给LLM的上下文====")
    print(context_text)
    answer = rag_chain.invoke({"context":context_text,"question":question})
    return {
        "code":0,
        "question":question,
        "answer":answer,
        "retrieve_docs":[ {"source":d.metadata["source"],"page_content":d.page_content} for d in docs ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)