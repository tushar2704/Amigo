##© 2024 Tushar Aggarwal. All rights reserved.(https://tushar-aggarwal.com)
##Amigo[Towards-GenAI] (https://github.com/Towards-GenAI)
##################################################################################################
#Importing dependencies
import datetime
import streamlit as st
from pathlib import Path
import base64
import sys
import os
import logging
import warnings
import asyncio
# loop = asyncio.new_event_loop()
# asyncio.set_event_loop(loop)
from dotenv import load_dotenv
from typing import Any, Dict
import google.generativeai as genai
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from crewai import Crew, Process, Agent, Task
from langchain_community.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()
#from src
from src.components.navigation import *
from src.crews.agents import *
from src.crews.tasks import *
from textwrap import dedent
import base64



#Homepage
page_config("Amigo", "🤖", "wide")
custom_style()
st.logo('./src/ygbj8rv2yafx6fsnqr2w.png')
st.sidebar.image('./src/ygbj8rv2yafx6fsnqr2w.png')
google_api_key = st.sidebar.text_input("Enter your GeminiPro API key:", type="password")

######################################################################################
#Intializing llm


llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True, 
                             temperature=0.2, google_api_key=google_api_key)
######################################################################################

# Custom Handler for logging interactions
class CustomHandler(BaseCallbackHandler):
    def __init__(self, agent_name: str) -> None:
        super().__init__()
        self.agent_name = agent_name

    def on_chain_start(self, serialized: Dict[str, Any], outputs: Dict[str, Any], **kwargs: Any) -> None:
        st.session_state.messages.append({"role": "assistant", "content": outputs['input']})
        st.chat_message("assistant").write(outputs['input'])

    def on_agent_action(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        st.session_state.messages.append({"role": "assistant", "content": inputs['input']})
        st.chat_message("assistant").write(inputs['input'])

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        st.session_state.messages.append({"role": self.agent_name, "content": outputs['output']})
        st.chat_message(self.agent_name).write(outputs['output'])
        
class AdvisorCrew:

    def __init__(self, origin, destination, interests , questions):
        self.origin = origin
        self.destination = destination
        self.interests = interests
        # self.problem = problem
        self.questions = questions
        self.answers = []


    def ask_questions(self):
        print("Please answer the following questions:")
        self.answers = []  # Clear previous answers if any
        for key, question in self.questions:
            if isinstance(question, str):
                answer = st.text_area(question + " ")

                self.answers.append((key,answer))
            else:
                print("Invalid question format. Skipping question.",key)

    def get_answers(self):
        return self.answers
 
    def run(self):
        agents = MoodMender()
        tasks = PsychTasks()

        syndrome_advisor_agent = agents.emotional_support()
        cultural_advisor_agent = agents.cultural_advisor()
        

        support_task = tasks.support_task(
        syndrome_advisor_agent,
        self.origin,
        self.destination,
        # self.problem,
        self.answers,
        )
        advisor_task = tasks.cultural_task(
        cultural_advisor_agent,
        self.origin,
        self.destination,
        # self.problem,
        self.interests
        )
        
        crew = Crew(
        agents=[
            syndrome_advisor_agent, cultural_advisor_agent
        ],
        tasks=[support_task, advisor_task],
        full_output=True,
        verbose=2,
        output_log_file=True,
        )

        result = crew.kickoff()
        return result

       













def main():
    st.title("🤖Amigo🤖")
    st.markdown('''
            <style>
                div.block-container{padding-top:0px;}
                font-family: 'Roboto', sans-serif; /* Add Roboto font */
                color: blue; /* Make the text blue */
            </style>
                ''',
            unsafe_allow_html=True)
    st.markdown(
        """
        ### Your Health Care Team, powered by Gemini Pro & CrewAI & [Towards-GenAI](https://github.com/Towards-GenAI)
        """
    )
    
    with st.container():
        st.header("👇 Enter your trip details")
        with st.form("my_form"):
            origin = st.text_input("Where are you from?")
                
            destination = st.text_input("Where are you currently in Paris?")
                
            interests = st.text_area("What are your interests?")
            
            questions = [
                        ("expectations","What were your expectations of Paris before you arrived? Were there specific places or experiences you were looking forward to in Paris?"),
                        ("experiences","How have your experiences in Paris differed from your expectations? Can you describe specific incidents or aspects of Paris that have been disappointing or distressing?"),
                        ("emotional_state","How are you feeling emotionally and mentally during your visit to Paris? Have you experienced symptoms like anxiety, depression, or disorientation since arriving in Paris?"),
                        ("social_interactions","How have your interactions with locals and other tourists been? Have you faced any language barriers or cultural misunderstandings?"),
                        ("coping_strategies","How are you currently coping with your feelings of disappointment or distress? Are there any activities or places in Paris that have helped improve your mood or alleviate stress?"),
                        ("support_systems","Do you have friends, family, or acquaintances in Paris you can talk to about your feelings? Are you in contact with any mental health professionals or support groups?"),
                        ("duration","How long do you plan to stay in Paris? Do you have any flexibility in your travel plans to visit other places or change your itinerary?"),
                        ("health_safety","Have you sought any medical or psychological assistance since experiencing these feelings? Do you feel safe and secure in your current accommodation and surroundings?"),
                        ("reflection","Looking back, is there anything you think could have prepared you better for this trip? What advice would you give to someone planning a trip to Paris to help them set realistic expectations?"),
                        ("personal_background","Is this your first time traveling abroad, or have you had similar experiences in other destinations? Can you share a bit about your general travel preferences and past travel experiences?"),
                    
                    ]
            advisor_crew = AdvisorCrew(origin, destination, interests, questions)
            advisor_crew.ask_questions()
            answers = advisor_crew.get_answers()
            # Count the number of tuples with non-empty second elements
            count_non_empty = sum(1 for key, value in answers if value)
            submitted = st.form_submit_button("Submit")

            
    # Check if any user input is empty before setting submitted
    
    
    if submitted and all([origin, destination, interests]) and count_non_empty >= 6:
        
        with st.status("🤖 **Agents at work...**", state="running", expanded=True) as status:
            with st.container(height=500, border=False):
                result = advisor_crew.run()
            status.update(label="✅ Advice is ready!",
                        state="complete", expanded=False)

        st.subheader("Here is the advice", anchor=False, divider="rainbow")
        
        task_outputs = result.get('tasks_outputs', [])
        headers = ['Mental Health Advice','Cultural Awareness']
        # Loop through each TaskOutput and display information
        for header, task_output in zip(headers, task_outputs):
            with st.expander(header):
                st.write(f"{task_output.raw_output}")
    
    
if __name__ == "__main__":
    main()
    with st.sidebar:
        footer()

