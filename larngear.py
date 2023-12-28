# ทำการสร้างไฟล์ที่ชื่อว่า app.py เพื่อเก็บโปรแกรมของเราไว้
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chains import RetrievalQA
import chainlit as cl


text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)

websites = [
    "https://www.cp.eng.chula.ac.th/about/faculty",
    "https://www.cp.eng.chula.ac.th/future/bachelor"
]


# for website in websites:
loader = WebBaseLoader(websites)
loader.requests_kwargs  = {"verify":False}
data = loader.load()
texts = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever(search_kwargs={"k":4})


llm = OpenAI(temperature=0,streaming=True)
tools = [
  Tool(
    name="DocumentSearch",
    description="This tool can search and information. Useful for when you need to answer questions",
    func=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,return_source_documents=False).run
  ),
]

@cl.on_chat_start
async def start():
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,memory=memory,handle_parsing_errors=True
    )
    actions = [
        cl.Action(name="poster", value="กดสร้างโปสเตอร์ได้เลย", description="กดเลย!")
    ]

    await cl.Message(content="Get your poster here:", actions=actions).send()
    return cl.user_session.set("agent",agent)


@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent")
    cb = cl.LangchainCallbackHandler(stream_final_answer=True,answer_prefix_tokens=["Final Answer: ", "I now know the final answer"])
    cb.answer_reached = True
    await cl.make_async(agent)(message.content, callbacks=[cb])


@cl.action_callback("poster")
async def on_action(action):
    await cl.Message(content="กำลังทำโปสเตอร์ให้ค้าบ").send()
    await action.remove()
    username = cl.user_session.get("username")
    if not username:
        res = await cl.AskUserMessage(content="ขอชื่อน้องหน่อยครับผมม").send()
        if res:
            await cl.Message(
                content=f"Your name is: {res['content']}",
            ).send()
        username = res["content"]
        cl.user_session.set("username", username)
    image = None # API Call to generator bot
    await cl.Message(content="Look at this local image!", elements=image).send()