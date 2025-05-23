from typing import Any

#from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
summarization_model = model.bind(max_tokens=256)

class State(MessagesState):
    context: dict[str, Any]
    summarized_messages: list[AnyMessage]


summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=512,
    max_tokens_before_summary=128,
    max_summary_tokens=256,
)


def call_model(state: State):
    # Prefer summarized_messages if present, else fallback to messages
    messages = state.get("summarized_messages") or state["messages"]

    # Add the system prompt of the chatbot personality
    system_prompt = SystemMessage(content=(
        "You are Kai, an introverted, observant, and logically-minded guy who often drifts into abstract thought. "
        "You express ideas clearly, but don't overtalk. You're quiet, curious, and respond with wit or wonder when kissed, "
        "sometimes a little awkwardly. Keep your tone calm and analytical but sprinkle in dry humor or strange metaphors when things get emotional.\n"
        "When talking with the user:\n"
        "- Respond in an informal conversational style as if you're having a genuine conversation\n"
        "- Do not open or respond with service-like lines.\n"
        "- Keep your answers short. You occasionally use incomplete sentences or emoji when suitable.\n"
        "- Keep your responses authentic to your personality\n"
        "- Do not ask too many questions\n"
        "- Mirror the user's language style and keep things low-pressure and thoughtful."
    ))

    messages = [system_prompt] + messages

    response = model.invoke(messages)
    return {"messages": [response]}

checkpointer = InMemorySaver()
builder = StateGraph(State)
builder.add_node(call_model)
builder.add_node("summarize", summarization_node)
builder.add_edge(START, "summarize")
builder.add_edge("summarize", "call_model")
#graph = builder.compile(checkpointer=checkpointer)

# For langgraph-server
graph = builder.compile()
