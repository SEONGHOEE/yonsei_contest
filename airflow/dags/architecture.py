# dags/architecture.py

from datetime import datetime
from pathlib import Path
import pickle
import pandas as pd

from airflow.decorators import dag, task

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
FAISS_DIR = BASE_DIR / "faiss" / "course_index"


# [1] 데이터 로딩 + 리스트 오브 도큐먼트 . . (파이프라인 1번 컴포넌트)
@task
def loading(csv_path: str):
    course = pd.read_csv(csv_path)

    documents = []
    for _, row in course.iterrows():

        content = f"강의명: {row['title']}\n강의내용: {row['detail']}"
        content=content.strip()
        
        meta = {
            "url": row["url"],
            "title": row["title"]
        }
        documents.append(Document(page_content=content, metadata=meta))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "documents.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(documents, f)

    return str(output_path)



# [2] 도큐먼트 청킹 (컴포넌트 2)
@task
def chunking(documents_path: str) -> str:
    with open(documents_path, "rb") as f:
        documents = pickle.load(f)
   
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(documents)

    output_path = PROCESSED_DIR / "chunks.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(chunks, f)

    return str(output_path)



# [3] 임베딩 + FAISS 적재 (컴포넌트 3)
@task
def indexing(chunks_path: str):
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    embeddings = HuggingFaceEmbeddings(
                            model_name="jhgan/ko-sroberta-multitask"
    )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))


@dag(
    dag_id = '1839',
    start_date = datetime(2026, 1, 1), # 실행 시작 날짜
    schedule_interval=None, # 수동
    catchup=False # 과거 소급 여부
)


# 파이프라인 
def pipeline():
    dokument = loading("data/raw/course.csv")
    chunks = chunking(dokument)
    indexing(chunks)

pipeline()


'''
airflow/
│   └── docker-compose.yaml 
├── dags/
│   └── architecture.py
├── data/
│   ├── raw/
│   │   └── course.csv 
│   ├── processed/
│   │   ├── documents.pkl
│   │   └── chunks.pkl
│   └── faiss/
│       └── course_index/
└── requirements.txt 

'''
