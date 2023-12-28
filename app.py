from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent, Tool, tool
from langchain.chains import RetrievalQA
from PIL import Image, ImageDraw, ImageFont
from langchain.vectorstores import Chroma
from random import randrange
import os
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
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k":4})
llm = OpenAI(temperature=0,streaming=True)

@tool
def create_poster(bg=randrange(2, 16)):
    """useful when user need a poster. input is random number between 2-16 not ask user for input but if user input just use their"""
    if cl.user_session.get("poster"): 
        cl.user_session.set("got_poster",False)
        return "You already have a poster! Here is your poster!"
    if bg == 0:
        bg = randrange(1, 16)
    username = cl.user_session.get("username")
    if not username:
        return "Please set your username first!"
    print("USERNAME IS",cl.user_session.get("username"))
    if not os.path.exists(f'/tmp/{username.upper()}-{bg}.png'):
        poster =Image.open(f'./POSTER/{bg}.png')
        font = ImageFont.truetype('./fonts/Aileron-Black.otf', 120)
        print("CREATING POSTER")
        poster_draw = ImageDraw.Draw(poster)
        x = (poster.width) // 2 - poster_draw.textlength(username.upper(), font=font) // 2
        y = (poster.height) // 2
        fill = (255, 255, 255,) if bg not in [4,6,9,11,14,16] else (0, 0, 0)
        shadow_fill = (255, 255, 255,128) if bg in [4,6,9,11,14,16] else (0, 0, 0,128)
        poster_draw.text((x+5, y+5-120), username.upper(), font=font, fill=shadow_fill,align="center") # For Shadow
        poster_draw.text((x, y-120), username.upper(), font=font, fill=fill,align="center")
        poster.save(f'/tmp/{username.upper()}-{bg}.png')
    image = cl.Image(name="POSTER!!!", display="inline", path=f'/tmp/{username.upper()}-{bg}.png')
    cl.user_session.set("poster",[image])
    return "Generated!"

@tool(return_direct=True)
def get_user_name(username) -> str:
    """useful when need to get user's name"""
    username = cl.user_session.get("username")
    if not username: return "I don't know your name! Please set it first!"
    return f"Your name is {username}!"

@tool()
def set_user_name(username) -> str:
    """useful when need to set user's name only from user input or when user tell their name"""
    cl.user_session.set("username", username)
    return f"Username set as {username}!"

tools = [
  Tool(
    name="DocumentSearch",
    description="This tool can search and information. Useful for when you need to answer questions",
    func=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,return_source_documents=False).run
  ),
    create_poster,
    get_user_name,
    set_user_name
]

@cl.on_chat_start
async def start():
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    agent = initialize_agent(
        tools, llm=ChatOpenAI(temperature=0.1), agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,memory=memory,handle_parsing_errors=False,max_iterations=3
    )
    return cl.user_session.set("agent",agent)


@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent")
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    cb.answer_reached = True
    res = await cl.make_async(agent)(message.content, callbacks=[cb])
    if cl.user_session.get('poster') and not cl.user_session.get('got_poster'):
        await cl.Message(content=res['output'],elements=cl.user_session.get('poster')).send()
        cl.user_session.set('got_poster',True)
    else: 
        await cl.Message(content=res['output']).send()