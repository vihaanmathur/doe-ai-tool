import streamlit as st
import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
import os
import shutil
import time

CHROMA_PATH = "chroma"
k = 5 #how many chunks we want to include in the prompt 
SIMILARITY_CUTOFF = 0.7 #How low of a similarity score we want to tell the user that no document has info relating to their question


#This is the modified RAG prompt, containing the additional context from our documents
#When we call the OpenAI API, we won't be just passing in the user's question
#It will be passed along with the k most relevant chunks from the database
#We can modify the prompt, to modify our output 
PROMPT_TEMPLATE = """
Answer the question in as much detail as possible based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def stream_output(text, delay=0.05):
    """
    Generator function that yields words from the input text one at a time with a delay.
    This is so we have the effect of typing from the AI. 
    
    Args:
    text (str): The input text to be streamed.
    delay (float): Delay in seconds between each word.
    
    Yields:
    str: The next word from the input text.
    """
    paragraphs = text.split('\n')
    for paragraph in paragraphs:
        words = paragraph.split()
        for word in words:
            yield word + ' '
            time.sleep(delay)  # Simulates streaming by adding delay
        yield '\n'  # Preserve paragraph structure


def main():
    st.title("ESS Insight AI") #The Title of our AI Tool 

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"]) #Gathering our OpenAI API Key so we can call the API 

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo" #Ensures the correct model is being called 

    with st.sidebar: 
        st.image('oe_logo.png', use_column_width="always")
        st.subheader('Chats       💬')
        st.button('Current Chat', use_container_width=True)
        st.button('Lithium-Ion Safety Discussion' , use_container_width=True)
        st.button('Compliance Standards for Transmission', use_container_width=True)
        st.button('FERC 2222 Questions', use_container_width=True)
        st.button('ESS Policy Analysis for California', use_container_width=True)
        st.subheader('')
        st.subheader('')
        st.subheader('Settings ⚙️')
        st.image('profile.png')

    '''
    The above lines under the "st.sidebar", put the OE logo as well as buttons representing the chats 
    to show the functionality of the real tool. At the bottom, we show the profile and settings icon. 
    Although they don't actually do anything here, it is good to include in the demo. 

    '''

    # Initialize chat history, as every past message in the chat is stores in the "session state"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Basically everytime a new chat is added, the website is reloaded with the new messages from the session state
    # The bottom few lines display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar = message["image"]): #An image attribute to ensure the correct images are displayed in chat history
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Type a message..."):
        # Display user message in chat message container
        st.chat_message("Vihaan", avatar='vihaan.png').markdown(prompt)

        #This is finding the Vector Database and putting it in the variable: db
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        #This uses LangChain, similarity search method to search through our database for the k most relevant chunks of data 
        results = db.similarity_search_with_relevance_scores(prompt, k)

        #The important feature below, allows us to filter out questions that aren't relevant or 
        #Let the user know that there is no information in the database that matches their question 
        if len(results) == 0 or results[0][1] < SIMILARITY_CUTOFF:
            no_match_text = "This prompt doesn't match any documents in the Energy Storage Database. Please enter a new prompt."
            st.session_state.messages.append({"role": "user", "content": prompt, "image":'vihaan.png'})
            with st.chat_message("assistant", avatar='doe_logo.png'):
                st.write_stream(stream_output(no_match_text))
                st.session_state.messages.append({"role": "assistant", "content": no_match_text, "image":'doe_logo.png'})
                '''
                The above code tells the user that they should enter in a new prompt, and ensures to add the messages to the 
                session state (chat history). 

                Chat Message Structure: {"role": either user or assistant, "content": the text, "image":the image to display next to the chat}
                '''
        else: 
            #The below code concatenates the original question along with the chunks of relevant information 
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            new_prompt = prompt_template.format(context=context_text, question=prompt)   
            
            # Add user message to chat history
            #We are ensuring to add in the modified prompt, so the Chat OpenAI Object, has the additional context 
            #That way, during the demo, we also get to see the prompt being sent to the LLM
            st.session_state.messages.append({"role": "user", "content": new_prompt, "image":'vihaan.png'})
            with st.chat_message("assistant", avatar='doe_logo.png'):
                response = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ], max_tokens=2048,
                temperature= 0.7 , 
                )
                #Above code is passing the entire message history to the LLM as its prompt 
                #This is so it keeps track of conversational history, and why any chat shouldn't last too long, as we might reach token limit
                sources = [f"{doc.metadata.get('source', None)} (Page {doc.metadata.get('page', None)})"
                                for doc, _score in results]
                sources_text = "\n\nSources:\n" + "\n".join(sources)
                message = response.choices[0].message.content
                formatted_response = f"{message} \n\n {sources_text}"
                st.write_stream(stream_output(formatted_response))
                st.session_state.messages.append({"role": "assistant", "content": formatted_response, "image":'doe_logo.png'})
                '''
                Above code modifies the LLM output by adding on the sources and then streaming it to the chat interface and adding the
                response to the session state. 
                '''


if __name__ == "__main__":
    main()