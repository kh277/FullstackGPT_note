# Chat GPT API - Document GPT




## 7.0강 - Introduction
이번 강의에서는 Streamlit을 이용하여 Document GPT를 배포하는 과정을 학습한다.  
6강에서 사용했던 Jupiter notebook이 아니라 Home.py라는 새로운 파일을 만들어 작성한다.  

``` python
import streamlit as st

st.title("Hello World!")

st.subheader("welcome to streamlit")
st.markdown("""i love it""")
```
위와 같이 작성하고 Terminal에
```
streamlit run Home.py
```
를 작성하고 실행하면 localhost에서 UI를 열 수 있게 해준다.  
즉, 내부에서 서버가 실행된다.  
만약 이 서버를 종료하고 싶으면 Ctrl(Command) + C를 입력하면 된다.  
