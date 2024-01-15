from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent, tool
from PIL import Image, ImageDraw, ImageFont
from random import randrange
import io
import chainlit as cl

@tool
def create_poster(bg=randrange(2, 16)):
    """useful when user need a poster. input is random number between 2-16 not ask user for input but if user input 1 just use their"""
    if cl.user_session.get("poster") and bg != 1: 
        cl.user_session.set("got_poster",False)
        return "You already have a poster! Here is your poster!"
    if not bg.isdigit():
        bg = randrange(2, 16)
    if 1 > int(bg) or 16 < int(bg):
        bg = randrange(2,16)
    username = cl.user_session.get("username")
    if not username:
        return "Please set your username first!"
    poster =Image.open(f'./POSTER/{bg}.png')
    font = ImageFont.truetype('./fonts/Aileron-Black.otf', 120)
    print(f"CREATING POSTER {username} - {bg}")
    poster_draw = ImageDraw.Draw(poster)
    x = (poster.width) // 2 - poster_draw.textlength(username.upper(), font=font) // 2
    y = (poster.height) // 2
    fill = (255, 255, 255,) if bg not in [4,6,9,11,14,16] else (0, 0, 0)
    shadow_fill = (255, 255, 255,128) if bg in [4,6,9,11,14,16] else (0, 0, 0,128)
    poster_draw.text((x+5, y+5-120), username.upper(), font=font, fill=shadow_fill,align="center") # For Shadow
    poster_draw.text((x, y-120), username.upper(), font=font, fill=fill,align="center")
    imgByteArr = io.BytesIO()
    poster.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    cl.user_session.set("poster",imgByteArr)
    return "Generated!"

@tool()
def get_user_name(username) -> str:
    """useful when need to get user's name"""
    username = cl.user_session.get("username")
    if not username: return "I don't know your name! Please set it by telling me your name first!"
    return f"Your name is {username}!"

@tool()
def set_user_name(username) -> str:
    """useful when need to set user's name only from user input or when user tell their name"""
    cl.user_session.set("username", username)
    return f"Username set as {username}!"

tools = [
    create_poster,
    get_user_name,
    set_user_name
]

@cl.on_chat_start
async def start():
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    agent = initialize_agent(
        tools, llm=ChatOpenAI(temperature=0.0,model="gpt-3.5-turbo"), agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,memory=memory,handle_parsing_errors=False,max_iterations=5
    )
    return cl.user_session.set("agent",agent)


@cl.on_message
async def main(message):
    if cl.user_session.get("TRY") == None:
        cl.user_session.set("TRY",0)
    if cl.user_session.get("TRY") >= 5:
        await cl.Message(content="Usage Limit at 5 messages :D").send()
    else:
        agent = cl.user_session.get("agent")
        cb = cl.LangchainCallbackHandler(stream_final_answer=True)
        cb.answer_reached = True
        res = await cl.make_async(agent)(message.content, callbacks=[cb])
        if cl.user_session.get('poster') and not cl.user_session.get('got_poster'):
            await cl.Message(content=res['output'],elements=[cl.Image(name="poster",display="inline",content=cl.user_session.get('poster'))]).send()
            cl.user_session.set('got_poster',True)
        else: 
            await cl.Message(content=res['output']).send()
        cl.user_session.set("TRY",cl.user_session.get("TRY") + 1)