# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str
    answer: str # 세율
    tax_base_equation: str # 과세표준 계산 수식
    tax_deduction: str # 공제액
    market_ratio: str # 공정시장가액비율
    tax_base: str # 과세표준 계산

graph_builder = StateGraph(AgentState)

# %%
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

text_path = "./documents/real_estate_tax.txt"

# 1) txt 파일을 Document 객체 리스트로 로드
loader = TextLoader(text_path, encoding="utf-8")
documents = loader.load()

# 2) 임베딩 함수
embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

# 3) 벡터스토어 생성 + 영구저장
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_function,          # ⚠️ embedding_function이 아니라 embedding
    collection_name="real_estate_tax",
    persist_directory="./real_estate_tax_collection"
)

# 4) Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # ⚠️ search_kwarg -> search_kwargs


# %% [markdown]
# 

# %%
query = '5억짜리 집 1채, 10억짜리 집 1채, 20억짜리 집 1채를 가지고 있을 때 세금을 얼마나 내나요?'

# %%
from langchain_classic import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

rag_prompt = hub.pull('rlm/rag-prompt')
llm = ChatOpenAI(model='gpt-4o')
small_llm = ChatOpenAI(model='gpt-4o-mini')

# %%
tax_base_retriever_chain = (
    {'context': retriever, 'question': RunnablePassthrough()} 
    | rag_prompt 
    | llm 
    | StrOutputParser()
)

tax_base_equation_prompt = ChatPromptTemplate.from_messages([
    ('system', '사용자의 질문에서 과세표준을 계산하는 방법을 수식으로 나타내주세요. 부연설명 없이 수식만 리턴해주세요.'),
    ('human', '{tax_base_equation_information}')
])
 
tax_base_equation_chain = (
    {'tax_base_equation_information': RunnablePassthrough()} 
    | tax_base_equation_prompt
    | llm 
    | StrOutputParser()
)

tax_base_chain = {'tax_base_equation_information':tax_base_retriever_chain} | tax_base_equation_chain

def get_tax_base_equation(state: AgentState):
    tax_base_equation_question = '주택에 대한 종합부동산세 계산시 과세표준을 계산하는 방법을 수식으로 표현해서 알려주세요.'
    tax_base_equation = tax_base_chain.invoke(tax_base_equation_question)
    return {'tax_base_equation': tax_base_equation}

# %%
get_tax_base_equation({})
# {'tax_base_equation': '과세표준 = (공시가격의 합계 - 공제금액) × 공정시장가액비율'}

# %%
tax_deduction_chain = (
    {'context': retriever, 'question': RunnablePassthrough()} 
    | rag_prompt 
    | llm 
    | StrOutputParser()
)

def get_tax_deduction(state: AgentState):
    tax_deduction_question = '주택에 대한 종합부동산세 계산시 공제금액을 알려주세요.'
    tax_deduction = tax_deduction_chain.invoke(tax_deduction_question)
    return {'tax_deduction': tax_deduction}

# %%
get_tax_deduction({})
# {'tax_deduction': '주택에 대한 종합부동산세 계산 시 공제 금액은 1세대 1주택자의 경우 12억 원, 법인 또는 법인으로 보는 단체의 경우 6억 원, 그리고 제1호 및 제2호에 해당하지 않는 일반적인 경우에는 9억 원입니다.'}

# %% [markdown]
# 웹 검색 부분 : 대통령령 

# %%
from langchain_community.tools import TavilySearchResults
from datetime import date

tavily_search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

tax_market_ration_prompt = ChatPromptTemplate.from_messages([
    ('system', f'아래 정보를 기반으로 공정시장 가액비율을 계산해주세요\n\nContext:\n{{context}}'),
    ('human', '{query}')
])

def get_market_ratio(state: AgentState):
    query = f'오늘 날짜:({date.today()})에 해당하는 주택 공시가격 공정시장가액비율은 몇%인가요?'
    context = tavily_search_tool.invoke(query)
    tax_market_ratio_chain = (
        tax_market_ration_prompt
        | llm
        | StrOutputParser()
    )
    tax_market_ratio = tax_market_ratio_chain.invoke({'context': context, 'query': query})
    return {'market_ratio': tax_market_ratio}

# %%
get_market_ratio({})
# {'market_ratio': '현재 주택 공시가격에 대한 공정시장가액비율은 60%입니다.'}

# %%
from langchain_core.prompts import PromptTemplate

tax_base_calculation_prompt = ChatPromptTemplate.from_messages(
    [
        ('system',"""
주어진 내용을 기반으로 과세표준을 계산해주세요

과세표준 계산 공식: {tax_base_equation}
공제금액: {tax_deduction}
공정시장가액비율: {market_ratio}"""),
        ('human', '사용자 주택 공시가격 정보: {query}')
    ]
)

def calculate_tax_base(state: AgentState):
    tax_base_equation = state['tax_base_equation']
    tax_deduction = state['tax_deduction']
    market_ratio = state['market_ratio']
    query = state['query']

    tax_base_calculation_chain = (
        tax_base_calculation_prompt
        | llm
        | StrOutputParser()
    )

    tax_base = tax_base_calculation_chain.invoke({
        'tax_base_equation': tax_base_equation,
        'tax_deduction': tax_deduction,
        'market_ratio': market_ratio,
        'query': query
    })
    print(f'tax_base == {tax_base}')
    return {'tax_base': tax_base}

# %%
initial_state = {
    'query': query,
    'tax_base_equation': '과세표준 = (공시가격의 합계 - 공제금액) × 공정시장가액비율',
    'tax_deduction': '주택에 대한 종합부동산세 계산 시 공제 금액은 1세대 1주택자의 경우 12억 원, 법인 또는 법인으로 보는 단체의 경우 6억 원, 그리고 제1호 및 제2호에 해당하지 않는 일반적인 경우에는 9억 원입니다.',
    'market_ratio': '현재 주택 공시가격에 대한 공정시장가액비율은 60%입니다.'
}

# %%
calculate_tax_base(initial_state)

# %%
tax_rate_calculation_prompt = ChatPromptTemplate.from_messages([
    ('system', '''당신은 종합부동산세 계산 전문가입니다. 아래 문서를 참고해서 사용자의 질문에 대한 종합부동산세를 계산해주세요

종합부동산세 세율:{context}'''),
    ('human', '''과세표준과 사용자가 소지한 주택의 수가 아래와 같을 때 종합부동산세를 계산해주세요

과세표준: {tax_base}
주택 수:{query}''')
])

def calculate_tax_rate(state: AgentState):
    query = state['query']
    tax_base = state['tax_base']
    context = retriever.invoke(query)
    tax_rate_chain = (
        tax_rate_calculation_prompt
        | llm
        | StrOutputParser()
    )
    tax_rate = tax_rate_chain.invoke({
        'context':context, 
        'tax_base': tax_base, 
        'query':query
    })
    print(f'tax_rate == {tax_rate}')
    return {'answer': tax_rate}

# %%
tax_base_state = {'tax_base': '주어진 정보를 바탕으로 과세표준을 계산해봅시다.\n\n1. 주택의 공시가격 합계:\n   - 5억 원짜리 집: 5억 원\n   - 10억 원짜리 집: 10억 원\n   - 20억 원짜리 집: 20억 원\n   - 공시가격 합계 = 5억 + 10억 + 20억 = 35억 원\n\n2. 공제금액:\n   - 입력하신 정보로는 1세대 1주택자가 아닌 것으로 보이므로 일반적인 경우에 해당합니다.\n   - 따라서 공제금액은 9억 원입니다.\n\n3. 공정시장가액비율:\n   - 60%입니다.\n\n과세표준 계산:\n- 과세표준 = (공시가격의 합계 - 공제금액) × 공정시장가액비율\n- 과세표준 = (35억 원 - 9억 원) × 60%\n- 과세표준 = 26억 원 × 0.6\n- 과세표준 = 15.6억 원\n\n따라서 과세표준은 15.6억 원입니다. 이 과세표준을 바탕으로 종합부동산세를 계산해야 합니다. 정확한 세액은 세율에 따라 달라지므로, 세율 정보를 확인하여 종합부동산세를 계산해야 합니다.', 'query':query}

# %%
calculate_tax_rate(tax_base_state)

# %%
graph_builder.add_node('get_tax_base_equation', get_tax_base_equation)
graph_builder.add_node('get_tax_deduction', get_tax_deduction)
graph_builder.add_node('get_market_ratio', get_market_ratio)
graph_builder.add_node('calculate_tax_base', calculate_tax_base)
graph_builder.add_node('calculate_tax_rate', calculate_tax_rate)

# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, 'get_tax_base_equation')
graph_builder.add_edge(START, 'get_tax_deduction')
graph_builder.add_edge(START, 'get_market_ratio')
graph_builder.add_edge('get_tax_base_equation', 'calculate_tax_base')
graph_builder.add_edge('get_tax_deduction', 'calculate_tax_base')
graph_builder.add_edge('get_market_ratio', 'calculate_tax_base')
graph_builder.add_edge('calculate_tax_base', 'calculate_tax_rate')
graph_builder.add_edge('calculate_tax_rate', END)

# %%
graph = graph_builder.compile()

