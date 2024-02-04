# Chat GPT API - Langchain 정리

## 3.4강 전체 코드

``` python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

# Chat model 설정
chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks = [StreamingStdOutCallbackHandler()]
)

# chef_chain, veg_chain에 대한 prompt 작성
chef_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world-class international chef. You create easy to follow \
      recipies for any type of cuosone with easy to find ingredients."),
    ("human", "I want to cook {cuisine} food.")
])

# langchain 생성
chef_chain = chef_prompt | chat
```

``` python
veg_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a vegetarian chef specialized on making traditional recipies vegetarian. \
      You find alternative ingredients and explain their preparation. You don't radically modify \
      the recipe. If there is no alternative for a food just say you don't know how to replace it."),
    ("human", "{question}")
])

veg_chain = veg_prompt | chat

# chef_chain의 결과물이 veg_chain의 recipe에 들어가야 하므로, 아래와 같이 작성함
# invoke를 2번 작성해도 되지만, 아래 코드가 더 보기 쉬움
final_chain = {"question" : chef_chain} | veg_chain

final_chain.invoke({
    "cuisine": "indian"
})

```

## 3.4강 Langchain input, output
#### Prompt -> Retriever -> Chat Model -> Tool -> Output Parser 순서로 진행.

1. Prompt
   
    + input type  : Dictionary
   
    + output type : PromptValue (즉, formatted prompt)

3. Chat model

    + input type  : 단일 String, chat message의 list, PromptValue
   
    + output type : ChatMessage (LLM 등 Chat AI model의 output)

5. Output Parser
   
    + input type  : ChatMessage


## 3.4강 Langchain input
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
final_chain = {"recipe" : chef_chain} | veg_chain
final_chain.invoke({
    "cuisine": "indian"
})
```
final_chain은 chef_chain과 veg_chain을 이어준다.

#### 결과적으로 {cuisine} --(chef_chain)--> {recipe} --(veg_chain)--> result가 된다.

``` python
final_chain = {"recipe" : chef_chain} | veg_chain
```

#### 이전 체인의 결과를 다음 체인으로 넘겨주려면 위와 같이 사용하면 된다.

invoke를 두 번 사용해도 되지만, 위의 코드가 더 간단하다.


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
