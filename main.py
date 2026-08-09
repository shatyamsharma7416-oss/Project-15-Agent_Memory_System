import os
from dotenv import load_dotenv
from openai import OpenAI
import datetime

from memory.short_term_memory import ShortTermMemory
from memory.long_term_memory import LongTermMemory

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
messages = [{"role": "system", "content": SYSTEM_PROMPT}] + stm.slide_chat + [{"role": "user", "content": user_input}]
while True:
    if user_input == "exit":
        ltm.add(stm.slide_chat)
        break

    print("\n\n"+str(messages)+"\n\n")
    response = client.chat.completions.create(
        model="auto",
        messages=messages,
        tools=[retrieve_facts_schema],
        tool_choice='auto'
    )
    # print(response.choices[0].message)

    if response.choices[0].message.content:
        print(response.choices[0].message.content)
        stm.add(user_msg=user_input, assistant_msg=response.choices[0].message.content)

    if response.choices[0].message.tool_calls:
        stm.add(user_msg=user_input, assistant_msg=response.choices[0].message.tool_calls)

        for tool_call in response.choices[0].message.tool_calls:
            tool_result = select_service(tool_call.function)
            print(tool_result)
            stm.add(assistant_msg={"role": "tool", "tool_call_id":tool_call.id, "content": str(tool_result)}, role="tool")

    if response.choices[0].finish_reason == "stop":
        user_input = input("Input your query: ")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + stm.slide_chat + [{"role": "user", "content": user_input}]



