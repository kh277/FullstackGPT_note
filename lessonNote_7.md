# Chat GPT API - Document GPT




## 7.0강 - Introduction
이번 강의에서는 Streamlit을 이용하여 Document GPT를 배포하는 과정을 학습한다.  
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





## 7.1강 - Magic
#### streamlit.write()
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


``` python
import streamlit as st
from langchain.prompts import PromptTemplate

st.selectbox("Choose your model", ("GPT-3", "GPT-4"))
```
위 코드는 selectbox를 만드는 코드이다.  
첫 번째 인자에는 label 즉 제목, 두 번째 인자에는 options 즉 선택할 수 있는 옵션이 들어간다.  
결과는 아래 사진과 같다.  
![image](https://github.com/kh277/test/assets/113894741/ddef40fd-237f-4748-a4e1-eb60a164e7db)


##
그 외에도 다른 API apperance를 보고 싶으면, 아래 링크에서 찾아보면 된다.
https://docs.streamlit.io/library/api-reference




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




# 7.3 - Multi Page
