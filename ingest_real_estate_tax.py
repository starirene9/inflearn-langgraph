"""
종합부동산세 문서를 real_estate_tax_chroma에 넣는 1회성 스크립트.
16.mcp_server.py 실행 전에 한 번만 실행하면 됩니다.
"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # .env에서 OPENAI_API_KEY 로드

from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# MCP 서버와 동일한 경로·컬렉션명 사용
PERSIST_DIR = "./real_estate_tax_chroma"
COLLECTION_NAME = "real_estate_tax"
DOC_PATH = "./documents/real_estate_tax.txt"

def main():
    if not Path(DOC_PATH).exists():
        print(f"문서가 없습니다: {DOC_PATH}")
        return
    loader = TextLoader(DOC_PATH, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(documents)
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    print(f"완료: {len(splits)}개 청크를 {PERSIST_DIR} ({COLLECTION_NAME})에 저장했습니다.")

if __name__ == "__main__":
    main()
