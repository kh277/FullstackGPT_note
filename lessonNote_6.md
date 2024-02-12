# Chat GPT API - RAG




## 6.0강 - Introduction
#### 전체적인 흐름 : <사용자의 질문> + <Vector에 저장된 질문과 관련된 문서들> 이 prompt에 추가되어 model로 입력됨.  
RAG의 첫 번째 단계 - Retrival(Langchain 모듈)  
  1. Load : 소스로부터 데이터 로드 (CSV, HTML, JSON, Markdown 등)
  2. Transform : Embedding하기 위해 데이터 분할
  3. Embed : 텍스트를 컴퓨터가 이해할 수 있는 숫자로 변환
  4. Store : Embedding한 데이터를 저장
  5. Retrieve




## 6.1강 - Load 과정 (Data Loader)

### TextLoader
``` python
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader

loader = TextLoader("./chapter_one.txt")
loader.load()
```
TextLoader는 txt 파일에서 텍스트를 읽어올 수 있다. 

### PyPDFLoader
``` python
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("./chapter_one.pdf")
loader.load()
```
PyPDFLoader는 pdf 파일에서 텍스트를 읽어올 수 있다.


### UnstructuredFileLoader
``` python
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader("./chapter_one.txt")
loader.load()
```
UnstructuredFileLoader는 텍스트 파일, 파워포인트, HTML, PDF, 이미지 등을 로드할 수 있다.  
즉, 파일 종류를 신경쓰지 않고 읽어올 수 있다.  
loader.load()를 할 경우 불러온 전체 문서가 Document로 묶여 list에 저장되는데, 이걸 분리할 필요가 있다.  




## 6.1강 - Transform 과정 (Splitter)
### RecursiveCharacterTextSplitter
``` python
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

loader = UnstructuredFileLoader("./chapter_one.txt")

loader.load_and_split(text_splitter=splitter)
```
RecursiveCharacterTextSplitter에서 chunk_size 옵션을 사용하면 청크 사이즈를 조절하면서 자를 수 있다.  
chunk_size 옵션만 넣으면 문장이 중간에서 잘릴 수 있기 때문에 chunk_overlap 옵션을 추가해 준다.  
위 두 옵션을 적용할 경우, 문장이 잘리면 앞쪽 Document에서 문장의 일부를 가져와 현재 Document에 추가하여 매끄럽게 해준다.  

### CharacterTextSplitter
``` python
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100
)

loader = UnstructuredFileLoader("./chapter_one.txt")

loader.load_and_split(text_splitter=splitter)
```
#### CharacterTextSplitter는 옵션에 separator를 추가할 수 있는데, 이는 python의 split()과 같은 역할을 한다.  




## 6.2강 - Transform 과정 (Tiktoken)
### from_tiktoken_encoder
```
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
    # length_function=len
)

loader = UnstructuredFileLoader("./chapter_one.txt")

loader.load_and_split(text_splitter=splitter)
```
CharacterTextSplitter의 옵션 중에 length_function이라는 기능이 있다.   
default값은 python의 내장 함수인 len()를 사용하도록 되어 있다.  
그러나 우리는 Chat GPT API를 사용하기 때문에 글자 개수가 아닌, token의 개수에 맞게 적용해야 한다.  
이것이 Tiktoken 패키지이다.




## 6.4강 - Embed, Store, Retrieve 과정 (Vector)
### Embedding
``` python
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore


# Load 과정
loader = UnstructuredFileLoader("./test.txt")


# Transform 과정
splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
)
docs = loader.load_and_split(text_splitter=splitter)


# Embed, Store 과정
# embedding 작업을 위한 embedder 선언
embeddings = OpenAIEmbeddings()

# embedding 작업 결과물을 저장할 캐시 저장소 설정
cache_dir = LocalFileStore("./.cache/")

# 캐시 저장소를 사용하는 embedding 생성
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

# 주어진 문서에 대한 Vector 생성
vectorstore = Chroma.from_documents(docs, cached_embeddings)

# Retrieve 과정
# vectorstore에서 예시 질의와 유사한 문서 Retrieve
results = vectorstore.similarity_search("where does winston live")
print(results)
```
위 코드에서 Load, Transform 과정은 6.2강에서 작성했던 코드와 동일하다.  
Embed, Store 과정은 아래와 같다.  
  1. 임베딩 작업을 위해 embedder를 OpenAIEmbeddings()를 이용하여 선언한다.  
  2. 임베딩 작업의 결과를 저장할 캐시 저장소를 LocalFileStore()를 이용하여 선언한다.  
  3. CacheBackedEmbeddings.from_bytes_store()를 이용하여 임베딩을 생성한다.  
      이 함수는 embedder와 캐시 저장소를 인자로 받는다.  
  4. Chroma.from_documents()를 이용하여 주어진 문서에 대한 Vector값을 생성한다.  
      이 함수는 Load, Transform 작업에서 생성한 Document, 3에서 생성한 임베딩을 인자로 받는다.
     
Retrieve 과정은 similarity_search()를 이용하여 캐시 저장소에 저장된 Vector값과 유사한 값을 찾는다.    
#### 챗봇을 통해 질문한 경우, Retrieve 작업을 통해 질문과 유사한 데이터 청크를 추출하고 prompt와 함께 LLM 모델로 넘겨주어 처리하도록 한다.  




## 6.6 RetrievalQA
#### Stuff Chain Method
``` python
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA

# LLM 선언
llm = ChatOpenAI()

loader = UnstructuredFileLoader("./test.txt")

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
)
docs = loader.load_and_split(text_splitter=splitter)

cache_dir = LocalFileStore("./.cache/")

embeddings = OpenAIEmbeddings()

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

vectorstore = Chroma.from_documents(docs, cached_embeddings)

# Stuff Chain 생성
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 응답 받아오기
chain.run("Where does Harry live? And Describe there.")
```
위 코드는 Stuff Document Chain을 사용한 코드이다.  
Stuff Chain은 질문을 모델에 요청할 때, 인자로 넘겨받은 Document 전체를 질문과 함께 prompt에 입력한다.  
따라서 prompt가 길어질 가능성이 있다.  
![image](https://github.com/kh277/test/assets/113894741/dadbc975-9831-40f0-a25e-69d85dd16857)





위 코드는 Refine Document Chain을 사용한 코드이다.
Refine Chain은 질문을 요청할 때, 각각의 Document를 읽으면서 질문에 대한 답변 생성을 시도한다.  
![image](https://github.com/kh277/test/assets/113894741/470c9a34-80bf-4a6a-857b-be63d58d37ae)

