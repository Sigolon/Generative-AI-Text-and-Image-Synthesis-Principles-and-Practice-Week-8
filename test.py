# Embedding
import pandas
import tqdm as notebook_tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document 

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)

# cot prompt + RAG
system_prompt_templete = '''
    你是一個優秀的資安分析師，現在需要幫助使用者分辨當前的情境屬於何種資安威脅。並參考威脅情資內容後，根據以下內容 Step by Step 分析。
    參考情資與使用者情境分析，
    1. 使用者可能是如何被入侵的
    2. 使用者目前可以先做何種緩解措施
'''

# RAG Post
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
vectorstore = FAISS.load_local(
    "data/faiss_db",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

messages = [
            {"role": "system", "content": system_prompt_templete}
            ]

import gradio
with gradio.Blocks() as demo:
    gradio.Markdown("# Professional Security Consulting")
    chatbot = gradio.Chatbot(type="messages")
    msg = gradio.Textbox(placeholder="請輸入你的問題...")
    state = gradio.State(messages)

    def main_chatbot(user_prompt, messages):
        results = vectorstore.similarity_search(user_prompt, k=1)
        RAG_Post = results[0].page_content
        user_prompt_templete = '''
            使用者情境
                {user_prompt}
            威脅情資  
                {RAG_Post}
        '''

        messages.append({"role": "user", "content": user_prompt_templete.format(user_prompt = user_prompt, RAG_Post= RAG_Post)})

        chat_completion = client.chat.completions.create(
            model= "gpt-4o-mini", # save money
            messages= messages,
            max_tokens= 500 # cot very expensive must be limit output token.
        )

        reply = chat_completion.choices[0].message.content
        messages.append({"role": "assistant", "content": reply}) # 透過添加歷史對話紀錄，變相讓 LLM 記得說了些什麼。改成這個寫法之後要 debug 也會比較方便。

        return "", messages, messages

    msg.submit(
            fn=main_chatbot,
            inputs=[msg, state],
            outputs=[msg, chatbot, state]
        )

demo.launch(share=True, debug=True)