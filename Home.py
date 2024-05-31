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
from langchain.agents import Tool
#from src
from src.components.navigation import *
from src.crews.agents import *
from src.crews.tasks import *
from textwrap import dedent
import base64



#Homepage
# page_config("Amigo", "🤖", "wide")
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
        


def create_crewai_setup(age, gender, disease):
        # Define Agents
        fitness_expert = Agent(
            role="Fitness Expert",
            goal=f"""Analyze the fitness requirements for a {age}-year-old {gender} with {disease} and 
                    suggest exercise routines and fitness strategies""",
            backstory=f"""Expert at understanding fitness needs, age-specific requirements, 
                        and gender-specific considerations. Skilled in developing 
                        customized exercise routines and fitness strategies.""",
            verbose=True,
            llm=llm,
            allow_delegation=True,
            tools=[search_tool],
        )
        
        nutritionist = Agent(
            role="Nutritionist",
            goal=f"""Assess nutritional requirements for a {age}-year-old {gender} with {disease} and 
                    provide dietary recommendations""",
            backstory=f"""Knowledgeable in nutrition for different age groups and genders, 
                        especially for individuals of {age} years old. Provides tailored 
                        dietary advice based on specific nutritional needs.""",
            verbose=True,
            llm=llm,
            allow_delegation=True,
        )
        
        doctor = Agent(
            role="Doctor",
            goal=f"""Evaluate the overall health considerations for a {age}-year-old {gender} with {disease} and 
                    provide recommendations for a healthy lifestyle.Pass it on to the
                    disease_expert if you are not an expert of {disease} """,
            backstory=f"""Medical professional experienced in assessing overall health and 
                        well-being. Offers recommendations for a healthy lifestyle 
                        considering age, gender, and disease factors.""",
            verbose=True,
            llm=llm,
            allow_delegation=True,
        )

        # Check if the person has a disease
        if disease.lower() == "yes":
            disease_expert = Agent(
                role="Disease Expert",
                goal=f"""Provide recommendations for managing {disease}""",
                backstory=f"""Specialized in dealing with individuals having {disease}. 
                            Offers tailored advice for managing the specific health condition.
                            Do not prescribe medicines but only give advice.""",
                verbose=True,
                llm=llm,
                allow_delegation=True,
            )
            disease_task = Task(
                description=f"""Provide recommendations for managing {disease}""",
                agent=disease_expert,
                llm=llm
            )
            health_crew = Crew(
                agents=[fitness_expert, nutritionist, doctor, disease_expert],
                tasks=[task1, task2, task3, disease_task],
                verbose=2,
                process=Process.sequential,
            )
        else:
            # Define Tasks without Disease Expert
            task1 = Task(
                description=f"""Analyze the fitness requirements for a {age}-year-old {gender}. 
                                Provide recommendations for exercise routines and fitness strategies.""",
                agent=fitness_expert,
                expected_output=f"""Recommendations for exercise routines and fitness strategies.""",
                llm=llm
            )

            task2 = Task(
                description=f"""Assess nutritional requirements for a {age}-year-old {gender}. 
                            Provide dietary recommendations based on specific nutritional needs.
                            Do not prescribe a medicine""",
                agent=nutritionist,
                expected_output=f"""Dietary recommendations based on specific nutritional needs.""",
                llm=llm
            )

            task3 = Task(
                description=f"""Evaluate overall health considerations for a {age}-year-old {gender}. 
                            Provide recommendations for a healthy lifestyle.""",
                agent=doctor,
                llm=llm, expected_output=f"""Recommendations for a healthy lifestyle.""",
            )
            
            health_crew = Crew(
                agents=[fitness_expert, nutritionist, doctor],
                tasks=[task1, task2, task3],
                verbose=2,
                process=Process.sequential,
            )

        # Create and Run the Crew
        crew_result = health_crew.kickoff()

        # Write "No disease" if the user does not have a disease
        if disease.lower() != "yes":
            crew_result += f"\n disease: {disease}"

        return crew_result


    
    

# Gradio interface
def run_crewai_app():
    st.subheader("Enter your details:")
    age = st.text_input("Age")
    gender = st.text_input("Gender")
    disease = st.text_input("Do you have any specific disease? (Enter 'Yes' or 'No')")
    
    if st.button("Submit"):
        if age and gender and disease:
            crew_result = create_crewai_setup(age, gender, disease)
            st.subheader("Health Recommendations:")
            st.write(crew_result)
        else:
            st.warning("Please fill in all the fields.")

    
if __name__ == "__main__":
    
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
    run_crewai_app()
    with st.sidebar:
        footer()

