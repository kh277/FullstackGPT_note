# Chat GPT API - Document GPT




## 7.0강 - Introduction
이번 7강에서는 Streamlit을 이용하여 Document GPT를 배포하는 과정을 학습한다.  
6강에서 사용했던 Jupiter notebook이 아니라 Home.py라는 새로운 파일을 만들어 작성한다.  

``` python
import streamlit as st

st.title("Hello World!")

st.subheader("welcome to streamlit")
st.markdown("""
  #### i love it
""")
```
위와 같이 작성하고 Terminal에
```
streamlit run Home.py
```
를 작성 후 실행하면 localhost에서 UI를 열 수 있게 해준다.  
즉, 내부에서 서버가 실행된다.  
만약 이 서버를 종료하고 싶으면 Ctrl(Command) + C를 입력하면 된다.  

<br>
<br>

## 7.1강 - Magic
### streamlit.write()
``` python
import streamlit as st
from langchain.prompts import PromptTemplate

st.write("hello")
st.write([1, 2, 3, 4])
st.write({"x":1})
st.write(PromptTemplate)
```
streamlit.write()는 입력한 내용을 사용자에게 보여주려고 한다.  
write()는 문자열, 리스트, 딕셔너리, 문서 등을 출력할 수 있다.  

<br>

##
### streamlit.selectbox()
``` python
import streamlit as st
from langchain.prompts import PromptTemplate

st.selectbox("Choose your model", ("GPT-3", "GPT-4"))
```
위 코드는 selectbox를 만드는 코드이다.  
첫 번째 인자에는 label 즉 제목, 두 번째 인자에는 options 즉 선택할 수 있는 옵션이 들어간다.  
결과는 아래 사진과 같다.  
![image](https://github.com/kh277/test/assets/113894741/ddef40fd-237f-4748-a4e1-eb60a164e7db)


<br>

##

그 외에도 다른 API apperance를 보고 싶으면, 아래 링크에서 찾아보면 된다.  
https://docs.streamlit.io/library/api-reference

<br>
<br>

## 7.2 - Data Flow
이 강의에서는 Streamlit의 Data flow와 데이터가 처리되는 방식을 정리한다.  
#### 결론부터 말하자면, 데이터가 변경될 때마다 python 파일 전체가 재실행된다.  
``` python
import streamlit as st
from datetime import datetime

today = datetime.today().strftime("%H:%M:%S")
st.title(today)

model = st.selectbox("Choose your model", ("GPT-3", "GPT-4"))
st.write(model)
```
![image](https://github.com/kh277/test/assets/113894741/535c5627-c89b-4afa-991a-58f44fd880a7)

위 코드를 실행시켜 보면, 실행할 때의 시각이 title에 나온다.  
이 상태에서 밑에 있는 옵션을 변경시키면, 변경시킨 시점의 시각으로 갱신된다.  
#### 즉, 무언가 단 하나라도 변경할 때마다 전체 파일이 재실행된다.  
이 부분은 React js, flutter처럼 새롭게 갱신된 부분만 refresh되는 웹과는 달리, 전체 페이지가 refresh된다.

<br>

##

``` python
import streamlit as st
from datetime import datetime

today = datetime.today().strftime("%H:%M:%S")
st.title(today)

name = st.text_input("what is your name?")
st.write(name)
```
위는 또다른 예시인데, 칸에 텍스트를 입력한 후 엔터키를 누를 때 refresh된다.  
#### 단, Streamlit에는 Cache를 제공하는 매커니즘이 있어서 어떤 것들은 다시 실행되지 않는다.  

<br>
<br>

## 7.3 - Multi Page
### st.sidebar()
``` python
import streamlit as st

st.title("title")

st.sidebar.title("sidebar title")
st.sidebar.text_input("xxx")
```
``` python
import streamlit as st

st.title("title")

with st.sidebar:
    st.title("sidebar title")
    st.text_input("xxx")
```
![image](https://github.com/kh277/test/assets/113894741/9c6241e7-505d-41b3-ba40-e95941bca8d9)
위의 두 코드는 전부 같은 기능을 하는 sidebar를 만드는 코드이다.  
가독성을 위해 2번째 코드처럼 작성하는 편이 좋다.  

<br>

##

``` python
import streamlit as st

st.title("title")

with st.sidebar:
    st.title("sidebar title")
    st.text_input("xxx")

tab_one, tab_two, tab_three = st.tabs(["A", "B", "C"])

with tab_one:
    st.write("a")

with tab_two:
    st.write("b")

with tab_three:
    st.write("c")
```
![image](https://github.com/kh277/test/assets/113894741/5ac9b071-4b3b-42db-b93f-5dff53df0b35)
위의 코드와 같이 with 키워드는 st.sidebar가 아닌 다른 곳에서도 사용할 수 있다.

<br>

##

``` python
import streamlit as st

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="✅"
)

st.title("FullstatGPT Home")
```
![image](https://github.com/kh277/test/assets/113894741/f94801b3-b0a8-415d-bdbd-696de465a199)  
위와 같이 작성할 경우, 결과사진과 같이 브라우저 탭의 이름과 아이콘이 변경된다.  

<br>

##
### pages 폴더
![image](https://github.com/kh277/test/assets/113894741/cfb1d822-9a8e-4bcd-a8a2-fb57ffa6add1)  
위 사진과 같이 pages라는 폴더를 만들어주고, 안쪽에 python 파일을 작성해 준다.

``` python
import streamlit as st

st.title("Document GPT")
```
![image](https://github.com/kh277/test/assets/113894741/ded26bb1-086a-4c73-b052-2b94a00b454a)  
위 코드는 사진의 pages라는 폴더 내의 01_DocumentGPT.py의 코드이다.  
결과화면과 같이 pages 폴더 내의 파일명은 sidebar에서 탭으로 표현된다.  

<br>
<br>

## 7.4 - Chat Messages
### streamlit.chat_message()
``` python
import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="✅"
)

st.title("DocumentGPT")

with st.chat_message("human"):
    st.write("Hello")

with st.chat_message("ai"):
    st.write("How are you")

with st.status("Embedding file...", expanded=True) as status:
    time.sleep(3)
    st.write("Getting the file")
    time.sleep(3)
    st.write("Embedding the file")
    time.sleep(3)
    st.write("Caching the file")
    status.update(lable="Error", state="error")

st.chat_input("send a message to the AI")
```
![image](https://github.com/kh277/test/assets/113894741/b426c790-b504-4d4e-869c-45bbad349725)  

st.chat_message()는 결과화면과 같이 AI-사람 간의 대화를 ChatGPT 형식처럼 출력해준다.  
또한 st.status()를 사용하면, 로딩과정을 표현할 수 있다.  

<br>

##
### example_1
``` python
import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="✅"
)

st.title("DocumentGPT")

def send_message(message, role):
    with st.chat_message(role):
        st.write(message)


message = st.chat_input("send a message to the AI")
if message:
    send_message(message, "human")
    time.sleep(2)
    send_message(f"You said : {message}", "ai")
```
![image](https://github.com/kh277/test/assets/113894741/1f3e0e01-e655-4142-8065-f5220f99e477)
위 코드는 사용자 입력을 받으면, 입력한 메시지를 자동으로 되돌려주는 챗봇이다.
#### 그러나 7.2강에서 설명한 Streamlit의 Data flow에 의해, 채팅을 새로 입력할 경우, 이전 메시지를 덮어 쓰는 문제가 발생한다.
위 문제를 해결하기 위해 이전 메시지들을 저장하는 저장소가 필요하다.  
단순하게 list를 선언해서 이전 메시지들을 누적시켜 저장한다고 생각할 수는 있지만, 문제는 코드가 항상 처음부터 끝까지 실행된다는 것이다.  
list, append를 사용하면 이전에 저장해둔 메시지들 또한 리셋이 되버린다.  
Streamlit의 Session_state는 refresh되지 않는다.  
이를 이용하면 chat 메시지를 저장할 수 있게 된다.

<br>

##
### example_2
``` python
import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="✅"
)

st.title("DocumentGPT")

# 메시지 저장소 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 메시지 저장
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})

# 이전 메시지 출력
for message in st.session_state["messages"]:
    send_message(message["message"], message["role"], save=False)

# 메시지 입출력
message = st.chat_input("send a message to the AI")
if message:
    send_message(message, "human")
    time.sleep(2)
    send_message(f"You said : {message}", "ai")

```
![image](https://github.com/kh277/test/assets/113894741/1b29a9ba-6a2a-444c-bda7-1cf7e247e1d5)

session_state를 추가한 코드이다.  
위 사진과 같이 이전 기록이 저장되어 있는 것을 확인할 수 있다.  

<br>
<br>

## 7.5 - Recap
Streamlit에서 사용자가 데이터를 변경할 때 코드 전체를 재실행한다.  
이 때, session_state를 제외하고 갱신하기 때문에 session_state는 데이터를 저장할 수 있는 저장소 역할을 한다.  
사용자가 처음 사용할 때 session_state는 비어있기 때문에 빈 리스트를 선언해주는 초기화 작업을 해줘야 한다.  

7.4절에서 선언된 함수 send_message는 2가지로 호출된다.  
첫째는 사용자가 메시지를 입력할 때이다.  
이 경우는 메시지를 session_state에 저장해야 하므로 send_message 함수를 그대로 호출하면 된다.  
두번재는 이전 메시지에 대한 기록을 불러와 write()할 때이다.  
이 경우는 session_state에 저장하면 안되므로 send_message에서 save=False라는 인자를 넘겨주어야 한다.  

<br>
<br>

##  7.6 - Uploading Documents
### example_1
``` python
import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="✅"
)

st.title("DocumentGPT")

st.markdown("""
    Welcome
""")

file = st.file_uploader(
    "upload a .txt .pdf or .docx file",
    type=["pdf", "txt", "docs"]
)

if file:
    st.write(file)
```
![image](https://github.com/kh277/test/assets/113894741/46e92953-1d99-4a01-97fa-9c7e9eb6de12)

위의 코드는 파일 업로더를 추가하여 파일을 입력받고, 정보를 출력해주는 기능을 한다.  
이제 해야 할 일은 6장에서 작성한 코드에 있는 UnstructedFileLoader에게 파일의 위치를 넘겨줘야한다.  
``` python
loader = UnstructuredFileLoader("./test.txt")
```

<br>

##

### example_2
``` python
import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="✅"
)

st.title("DocumentGPT")

st.markdown("""
    Welcome
""")

file = st.file_uploader(
    "upload a .txt .pdf or .docx file",
    type=["pdf", "txt", "docs"]
)

if file:
    st.write(file)
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    st.write(file_content, file_path)
    with open(file_path, "wb") as f:
        f.write(file_content)
```
![image](https://github.com/kh277/test/assets/113894741/5b151e83-24aa-48a7-b156-ffdc851857dc)  
추가된 내용은 다음과 같다.  
file.read()로 파일에 대한 정보를 읽고, st.write()로 파일의 내용을 화면에 출력한다.  
그 뒤, 파일을 f.write()를 이용하여 ./.cache/files 위치에 저장한다.  

<br>

##
### example_3
``` python
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="✅"
)

st.title("DocumentGPT")

st.markdown("""
    Welcome
""")

file = st.file_uploader(
    "upload a .txt .pdf or .docx file",
    type=["pdf", "txt", "docs"]
)

if file:
    st.write(file)
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    st.write(file_content, file_path)
    with open(file_path, "wb") as f:
        f.write(file_content)
    # 캐시 저장소 위치
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    # Splitter 선언
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader("./.cache/files/chapter_one.txt")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    docs = retriever.invoke("ministry of truth")
    st.write(docs)
``` 
6장에서 작성한 코드 중 일부를 가져와 합쳤다.  
그리고 캐시 저장소와 입력받은 파일을 저장하기 위한 위치를 바꾸었다. 
또한, Jupiter Notebook에서는 .env 파일 내에 API_KEY를 넣으면 자동으로 인식했지만, Streamlit은 그렇지 않다.  
따라서 .streamlit이라는 폴더를 생성한 후 그 안에 secret.tomld이라는 파일에 API_KEY를 넣었다.  
<br>
전체적인 파일 구성은 이렇게 된다.  
![image](https://github.com/kh277/test/assets/113894741/0038d851-d5de-4bea-88ea-a4014bffd6b9)  
입력받은 파일은 사진의 .cache/files 내에, 캐시 저장소는 .cache/embedding 내에 저장될 것이다.

![image](https://github.com/kh277/test/assets/113894741/378941e8-249c-4884-945d-3dde7cd8ff97) 
결과는 다음과 같다.  
해당 텍스트 파일을 읽어 "harry"라는 내용과 관련이 있는 문서 청크를 반환한다.  

<br>

##
### example_4
``` python
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="✅"
)


def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    # 캐시 저장소 위치
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    # Splitter 선언
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever


st.title("DocumentGPT")

st.markdown("""
    Welcome
""")

file = st.file_uploader(
    "upload a .txt .pdf or .docx file",
    type=["pdf", "txt", "docs"]
)

if file:
    retriever = embed_file(file)
    s = retriever.invoke("Harry")
    st.write(s)
```
example_3을 함수로 모듈화하여 좀 더 매끄럽게 굴러가도록 했다.  
그러나 사용자가 챗봇에 입력할 때마다 이 연산을 반복해야 하므로 문제가 발생할 수 있다.  


<br><br>


## 7.7 - Chat History
이전 강의에서 연산이 오래걸리는 embed_file 함수를 반복적으로 호출하는 문제점이 있었다.  
이를 해결하기 위해 embed_file 함수 윗부분에 아래 decorator를 추가해준다.  
``` python
@st.cache_data(show_spinner="Embedding file...")
```
이 decorator는 Streamlit이 함수를 실행하기 전에 어떤 파일이 있는지부터 확인한다.  
streamlit은 파일을 해싱한 뒤 동일한 파일인지 체크한다.
만약 파일이 동일하다면, 그 함수는 실행시지키 않고 기존에 반환한 값을 다시 반환한다.  
위 코드에 decorator를 추가하여 실행하면 아래 사진과 같게 나온다.  
![image](https://github.com/kh277/test/assets/113894741/89c06202-e9eb-4560-9a51-c41cb3a5ec9c)
처음 한번만 저 spinner가 나오고 다음부터는 나오지 않는다.  

<br>

## example_1
``` python
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="✅"
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    # 캐시 저장소 위치
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    # Splitter 선언
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    # 문서 Load
    loader = UnstructuredFileLoader(file_path)

    # 문서 Embed
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()

    # 문서 Cache
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    
    # vectorstore에 embedding ㄶ음
    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    # retriever 생성
    retriever = vectorstore.as_retriever()

    return retriever


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False
        )


st.title("DocumentGPT")

st.markdown("""
    Welcome
""")

with st.sidebar:
    file = st.file_uploader(
        "upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docs"]
    )

if file:
    retriever = embed_file(file)
    send_message("I'm Ready. Ask away", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        send_message("test", "ai")
else:
    st.session_state["messages"] = []
```
이전 채팅 기록을 볼 수 있는 기능을 추가되었다.  
session_state 초기화, 파일 초기화 시 채팅기록 초기화 등 세부 기능도 추가되었다.  
![image](https://github.com/kh277/test/assets/113894741/3d58e4d0-c7a0-4b45-989d-c3b7d91b832d)


<br><br>


## 7.8 - Chain
### example_1
``` python
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="✅"
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

llm = ChatOpenAI(
    temperature=0.1
)

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    # 캐시 저장소 위치
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    # Splitter 선언
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    # 문서 Load
    loader = UnstructuredFileLoader(file_path)

    # 문서 Embed
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()

    # 문서 Cache
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    
    # vectorstore에 embedding ㄶ음
    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    # retriever 생성
    retriever = vectorstore.as_retriever()

    return retriever


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer the question using only the following context. If you don't know the answer
     just say you don't know. DON'T make anything up.
     Context: {context}"""),
    ("human", "{question}")
])

st.title("DocumentGPT")

st.markdown("""
    Welcome
""")

with st.sidebar:
    file = st.file_uploader(
        "upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docs"]
    )

if file:
    retriever = embed_file(file)
    send_message("I'm Ready. Ask away", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")

        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        } | prompt | llm
        response = chain.invoke(message)
        send_message(response.content, "ai")
else:
    st.session_state["messages"] = []
```


<br><br>


## 7.9 - Streaming
### Callback Handler
이번에는 LLM에서 응답을 생성할 때, 완성되면 출력하는 것이 아니라 실시간으로 출력하도록 변경해볼 것이다.  
``` python
# ...

class ChatCallbackHandler(BaseCallbackHandler):  # --- 추가된 함수
    message = ""
    
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        with st.sidebar:
            st.write("llm ended")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


if "messages" not in st.session_state:
    st.session_state["messages"] = []

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,  # -- 추가
    callbacks=[ChatCallbackHandler()]  # --- 추가
)

# ...

```
위의 코드가 추가된 부분이다.  
1. llm이 시작되면(on_llm_start), empty_box를 message_box 안에 저장한다.  
2. 그 뒤, 토큰을 생성하여 받으면(on_llm_new_token) message_box에 token을 추가한다.  
3. 생성을 완료하면(on_llm_end), sidebar에 llm ended라는 문장이 추가된다.
   
아래 사진은 그 결과이다.

![image](https://github.com/kh277/test/assets/113894741/fbb670a0-6339-479e-bb99-3f0ec5872835)  

<br>

##
### fix_1
위 코드는 응답을 실시간으로 보여주긴 하지만, AI가 말하는 것처럼 보이지 않는다.  
<br>

``` python
llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()]
```
위 코드는 llm 선언 부분이고,  

``` python
if file:
    retriever = embed_file(file)
    send_message("I'm Ready. Ask away", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")

        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )
        response = chain.invoke(message)
        send_message(response.content, "ai")
else:
    st.session_state["messages"] = []
```
위 코드는 7.8에 있는 코드의 응답 생성 부분이다.  
7.9에 ChatCallbackHandler의 함수들은 chain.invoke()가 실행될 때에 호출되므로 해당 부분을 바꿔주어야 한다.

<br>

``` python
if file:
    retriever = embed_file(file)
    send_message("I'm Ready. Ask away", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")

        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):  # --- 추가
            response = chain.invoke(message)  # --- 한칸 들여쓰기
        # send_message(response.content, "ai") --- 삭제
else:
    st.session_state["messages"] = []
```
response 부분을 with st.chat_message("ai")로 감싸주면,  
CallbackHandler가 st.empty 함수를 호출할 때(on_llm_start), 토큰을 추가할 때(on_llm_new_token) AI가 한 것처럼 보일 것이다.  
이대로 작성하게 되면 실시간으로 답변을 생성하여 보여주고, 다 출력되면 send_message()에 의해 답변이 한번 더 출력될 것이다.  
2번 출력할 필요는 없으므로 send_message(response.content, "ai") 해당 부분은 지워주면 된다.  

<br>

##

send_message()를 삭제하면 cache에 저장할 수 없게 되므로 send_message() 부분을 on_llm_end가 호출될 때 메시지를 저장하도록 변경한다.  

``` python
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")  # --- 수정

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

# ...

def save_message(message, role):  # --- 추가된 함수
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)  # --- 수정
```

![image](https://github.com/kh277/test/assets/113894741/636b38eb-2b6b-41ba-b8c6-87ddb6894c0a)
메시지도 한번만 출력이 되면서 이전 메시지에 대한 기록도 남아있게 된다.  


<br><br>


## 7.10 - Recap
![image](https://github.com/kh277/test/assets/113894741/5e47f196-0007-42d5-8613-ccb19cd5119e)  

``` python
from typing import Dict, List
from uuid import UUID
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="✅"
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


if "messages" not in st.session_state:
    st.session_state["messages"] = []

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()]
)

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    # 캐시 저장소 위치
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    # Splitter 선언
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    # 문서 Load
    loader = UnstructuredFileLoader(file_path)

    # 문서 Embed
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()

    # 문서 Cache
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    
    # vectorstore에 embedding 넣음
    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    # retriever 생성
    retriever = vectorstore.as_retriever()

    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer the question using only the following context. If you don't know the answer
     just say you don't know. DON'T make anything up.
     Context: {context}"""),
    ("human", "{question}")
])

st.title("DocumentGPT")

st.markdown("""
    Welcome
""")

with st.sidebar:
    file = st.file_uploader(
        "upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docs"]
    )

if file:
    retriever = embed_file(file)
    send_message("I'm Ready. Ask away", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")

        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)
else:
    st.session_state["messages"] = []
```
파일 구성과 전체 코드이다.  
한 줄씩 찬찬히 살펴보면,  

``` python
# 1. 기본 설정

# 1-1. 제목 설정
st.title("DocumentGPT")

# 1-2. 제목 밑의 문구 설정
st.markdown("""
    Welcome
""")

# 1-3. 사이드바 설정 - 사용자에게 파일 요청
with st.sidebar:
    file = st.file_uploader(
        "upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docs"]
    )

# 2. 파일이 존재한다면(사용자가 파일을 업로드했다면),
if file:
    # 2-1. 파일을 embed해서 retriever 반환
    retriever = embed_file(file)

    # 2-2. 준비가 완료됐다는 메시지 전송
    send_message("I'm Ready. Ask away", "ai", save=False)

    # 2-3. 이전 메시지에 대한 기록 출력
    paint_history()

    # 2-4. chat input 생성
    message = st.chat_input("Ask anything about your file...")

    # 2-5. 만약 메시지를 보낸다면,
    if message:
        # 메시지를 human으로 출력
        send_message(message, "human")

        # langchain 실행
        chain = (
            {
                # retriever가 document의 리스트를 제공하면 format_docs() 함수를 거쳐 하나의 string으로 통합
                "context": retriever | RunnableLambda(format_docs),

                # 사용자가 질문하면 수정하지 않고 즉시 prompt로 전송
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )

        # ai 응답 생성
        with st.chat_message("ai"):
            chain.invoke(message)
# 2-0. 파일이 존재하지 않는다면,
else:
    # session_state에 저장된 메시지 초기화
    st.session_state["messages"] = []
```

<br>

2-1 과정의 embed_data() 함수
``` python
# 0. 데코레이터 - 동일한 파일에 대해 함수 재실행 방지
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    # 1. 파일을 읽음
    file_content = file.read()
    
    # 2. 파일을 복사하여 해당 위치에 저장함
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 3. splitter, loader 생성
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)

    # 4. document split 과정
    docs = loader.load_and_split(text_splitter=splitter)

    # 5. cache에서 임베딩 가져오기
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    
    # 6. document와 임베딩으로부터 vectorstore 획득
    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    # 7. vectorstore를 retriever로 변경
    retriever = vectorstore.as_retriever()

    return retriever
```

<br>

2-2 과정의 send_message() 함수
``` python
def send_message(message, role, save=True):
    with st.chat_message(role):
        # message 출력
        st.markdown(message)
    if save:
        # save 옵션이 켜져있다면, 메시지 저장
        save_message(message, role)

def save_message(message, role):
    # 메시지 딕셔너리를 저장하여 누가 보냈는지와 메시지의 내용 저장
    st.session_state["messages"].append({"message": message, "role": role})
```

<br>

2-3 과정의 paint_history() 함수
``` python
def paint_history():
    # session_state에 존재하는 모든 메시지에 대해
    for message in st.session_state["messages"]:
        # send_message 함수를 통해 출력
        send_message(
            message["message"],
            message["role"],
            save=False
        )
```

<br>

``` python
# BaseCallbackHandler는 llm에서 어떤 일이 발생하면 모든 method와 class 호출
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    # 신경쓸 사항 1 (llm 시작) - 화면에 empty_box 생성
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    # 신경쓸 사항 2 (llm 종료) - 메시지를 session_state에 저장
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    # 신경쓸 사항 3 (llm에서 새로운 token 추가) - empty_box 업데이트
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)
```
