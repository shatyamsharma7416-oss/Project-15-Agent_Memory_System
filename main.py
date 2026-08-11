import os
from dotenv import load_dotenv
from openai import OpenAI
import datetime

from memory_code import ShortTermMemory
from memory_code import LongTermMemory

from tools.facts_retrieve import retrieve_facts_schema
from function_call import select_service
from prompts import SYSTEM_PROMPT

load_dotenv()
client = OpenAI(
    base_url="https://freellmapi-seyc.onrender.com/v1",
    api_key=os.environ.get("FREE_LLM_API")
)


stm = ShortTermMemory(mode ="summary_buffer", llm_client=client)
ltm = LongTermMemory(llm_client=client)


session_id = ltm.session_id

user_input = input("Input your query: ")
stm.add(assistant_msg={"role": "user", "content": user_input}, role="tool")
messages = [{"role": "system", "content": SYSTEM_PROMPT}] + stm.slide_chat
while True:
    if user_input.strip().lower() == "exit":
        ltm.add(stm.slide_chat)
        break

    print("\n\n"+str(messages[1:]))
    response = client.chat.completions.create(
        model="auto",
        messages=messages,
        tools=[retrieve_facts_schema],
        tool_choice='auto'
    )
    msg = response.choices[0].message

    print(response.choices[0].message)


    if msg.tool_calls:
        assistant_tool_msg = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        }
        stm.add(assistant_msg=assistant_tool_msg, role="tool")

        for tool_call in msg.tool_calls:
            tool_result = select_service(tool_call.function)
            stm.add(
                assistant_msg={
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result),
                },
                role="tool",
            )
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + stm.slide_chat

    if response.choices[0].finish_reason == "stop":
        stm.add(assistant_msg={"role": "assistant", "content": msg.content}, role="tool")
        user_input = input("Input your query: ")
        stm.add(assistant_msg={"role": "user", "content": user_input}, role="tool")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + stm.slide_chat
