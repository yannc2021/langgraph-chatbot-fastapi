import asyncio
import httpx

API_URL = "http://127.0.0.1:8000/chat"
THREAD_ID = "1"

async def test_chat_sequence():
    async with httpx.AsyncClient(timeout=60.0) as client:
        messages = [
            "hi, my name is alice. I live in Nanjing",
            "Tell me what is the meaning of life",
            "Also tell me the meaning of physics",
            "What's the weather like in Nanjing today?",
            "Can you write a short poem about the Yangtze River?",
            "Who is the president of China?",
            "Summarize what we've talked about so far.",
            "What is the capital of France?",
            "Tell me a joke about cats.",
            "What did I say about my name?",
            "What is the population of Nanjing?",
            "Remind me where I live.",
            "What's my favorite subject?",
            "Summarize our conversation again."
        ]
        last_response = None
        for i, msg in enumerate(messages, 1):
            payload = {"messages": msg, "thread_id": THREAD_ID}
            resp = await client.post(API_URL, json=payload)
            resp.raise_for_status()
            last_response = resp.json()
            print(f"\nMessage {i}: {msg}")
            print("Response:", last_response.get("response"))
            print("Summary:", last_response.get("summary"))

if __name__ == "__main__":
    asyncio.run(test_chat_sequence())