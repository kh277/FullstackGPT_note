# Chat GPT API - Langchain 정리



## 4.1강 전체 코드

``` python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate


# Chat model
chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)


# 예제 작성
examples = [
    {
        "question": "What do you know about France?",
        "answer": """
        Here is what I know:
        Capital: Paris
        Language: French
        Food: Wine and Cheese
        Currency: Euro
        """
    },
    {
        "question": "What do you know about Italy?",
        "answer": """
        Here is what I know:
        Capital: Rome
        Language: Italian
        Food: Pizza and Pasta
        Currency: Euro
        """
    },
    {
        "question": "What do you know about Greece?",
        "answer": """
        Here is what I know:
        Capital: Athens
        Language: Greek
        Food: Souvlaki and Feta Cheese
        Currency: Euro
        """
    }
]


# 예제 형식화
example_template ="""
    Hunan: {question}
    AI: {answer}
"""

example_prompt = PromptTemplate.from_template("Human: {question}\nAI: {answer}")


# FewShotPromptTemplate에게 전달
prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    suffix="Human: What do you know about {country}?",
    input_variables=["country"]
)


# Langchain 작성
chain = prompt | chat
chain.invoke({
    "country": "Germany"
})

```



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



## 4.3강 전체 코드
```
from langchain.chat_models import ChatOpenAI
from langchain.prompts import example_selector
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector


# Chat model
chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

examples = [
    {
        "question": "What do you know about France?",
        "answer": """
        Here is what I know:
        Capital: Paris
        Language: French
        Food: Wine and Cheese
        Currency: Euro
        """,
    },
    {
        "question": "What do you know about Italy?",
        "answer": """
        I know this:
        Capital: Rome
        Language: Italian
        Food: Pizza and Pasta
        Currency: Euro
        """,
    },
    {
        "question": "What do you know about Greece?",
        "answer": """
        I know this:
        Capital: Athens
        Language: Greek
        Food: Souvlaki and Feta Cheese
        Currency: Euro
        """,
    },
]


# 랜덤 예시를 선택하는 클래스
class RandomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)
    
    def select_examples(self, input_variables):
        from random import choice
        
        return [choice(self.examples)]
    

example_prompt = PromptTemplate.from_template("Human: {question}\nAI:{answer}")

example_selector = RandomExampleSelector(
    examples=examples,
    # max_length=10
)

prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    suffix="Human: What do you know about {country}?",
    input_variables=["counrty"]
)

prompt.format(country="Brazil")

```

위 코드는 LengthBasedExampleSelector를 사용한 예제이다.  
예시에 너무 많은 글자가 들어갈 경우 이를 제한할 수 있다.