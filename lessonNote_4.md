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
chef_prompt = ChatPromptTemplate.from_messages([
    ("system", "system setting1"),
    ("human", "I want to cook {cuisine} food.")
])

veg_chef_prompt = ChatPromptTemplate.from_messages([
    ("system", "system setting2"),
    ("human", "{recipe}")
])

chef_chain = chef_prompt | chat
veg_chain = veg_chef_prompt | chat
```

위 코드에서,  
chef_chain의 입력값은 cuisine이고, veg_chain의 입력값은 recipe이다.  
또한, chef_chain의 출력값이 veg_chain의 입력값인 recipe로 들어간다.    

``` python
final_chain = chef_chain | veg_chain
final_chain.invoke({
    "cuisine": "indian"
})
veg_chain.invoke({
    "recipe": "chatmodel"
})
```



``` python
final_chain = {"recipe" : chef_chain} | veg_chain
final_chain.invoke({
    "cuisine": "indian"
})
```
위의 두 코드는 final_chain에 대한 설명이다.  
final_chain은 chef_chain과 veg_chain을 이어주는데, 두 가지 방법으로 구현할 수 있다.

첫 번째 코드는 invoke를 두 번 사용해서 결과를 출력하는 방법이다.

두 번째 코드는 invoke를 한 번만 사용하되, 체인을 생성하는 과정에서 인자를 바로 넘겨주는 방법이다.  
final_chain은 chef_chain과 veg_chain을 이어준다.  
#### 결과적으로 {cuisine} --(chef_chain)--> {recipe} --(veg_chain)--> result가 된다.
#### 이전 체인의 결과를 다음 체인으로 넘겨주려면 마지막 줄 같이 사용하면 된다.  


## 3.5강 Recap
``` python
chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks = [StreamingStdOutCallbackHandler()]
)
```

3.4강에서 작성했던 코드 중 Chat model 설정은 위와 같이 작성된다.  
temperature 은 0~1 사이의 값이며, 응답의 창의성을 조절한다.  
streaming은 Chat model의 응답이 생성될 때마다 즉시 얻을 수 있게 해준다.  
callback은 볼 수 있는 문자가 생길 때마다 console로 print를 해준다.  
callback은 LLM이 작업을 시작했을 때나, 작업을 끝냈을 때, 문자를 생성했을 때, 에러가 발생했을 때 등 이벤트에 대해 반응할 수도 있다.  
