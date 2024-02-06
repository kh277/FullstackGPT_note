# Chat GPT API - RAG

## 6.0강 - Introduction
#### 전체적인 흐름 : <사용자의 질문> + <Vector에 저장된 질문과 관련된 문서들> 이 prompt에 추가되어 model로 입력됨.  
RAG의 첫 번째 단계 - Retrival(Langchain 모듈)  
  1. Load : 소스로부터 데이터 로드 (CSV, HTML, JSON, Markdown 등)
  2. Transform : Embedding하기 위해 데이터 분할
  3. Embed : 텍스트를 컴퓨터가 이해할 수 있는 숫자로 변환
  4. Store : Embedding한 데이터를 저장
  5. Retrieve

## 6.1강 - Data Loader과 Splitter

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
CharacterTextSplitter는 옵션에 separator를 추가할 수 있는데, 이는 python의 split()과 같은 역할을 한다.  













## 4.1강 FewShotPromptTemplate 사용 예제
#### 예제를 제시하고 해당 예시에 맞게 출력 형식화하기 

1. 예제 작성하기   
2. Prompt를 사용해서 예제 형식화하기   
3. FewShotPromptTemplate에게 전달하기
   + example_prompt는 예제를 형식화 함
   + examples는 각각의 예제를 가져옴
   + suffix는 내용 마지막에 질문을 넣어줌



## 4.2강 전체 코드
``` python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import ChatMessagePromptTemplate, ChatPromptTemplate

# Chat model
chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

examples = [
    {
        "country": "France",
        "answer": """
        I know this:
        Capital: Paris
        Language: French
        Food: Wine and Cheese
        Currency: Euro
        """
    },
    {
        "country": "Italy",
        "answer": """
        I know this:
        Capital: Rome
        Language: Italian
        Food: Pizza and Pasta
        Currency: Euro
        """
    },
    {
        "country": "Greece",
        "answer": """
        I know this:
        Capital: Athens
        Language: Greek
        Food: Souvlaki and Feta Cheese
        Currency: Euro
        """
    }
]


example_prompt = ChatPromptTemplate.from_messages([
        ("human", "What do you know about {country}?"),
        ("ai", "{answer}")
])

# 응답 형식화
example_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a geography expert."),
        example_prompt,
        ("human", "What do you know about {country}?")
])

chain = final_prompt | chat
chain.invoke({"country": "Germany"})
```

위의 코드는 4.1과 같은데 FewShotPromptTemplate 대신 FewShotChatMessagePromptTemplate를 사용한 것이다.  
이를 통해 chatbot에서 사용하는 형태의 응답을 만들 수 있다.  




