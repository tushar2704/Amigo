##© 2024 Tushar Aggarwal. All rights reserved.(https://tushar-aggarwal.com)
##MoodMender[Towards-GenAI] (https://github.com/Towards-GenAI)
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
from crewai import Task
from textwrap import dedent
from datetime import date

class PsychTasks():

    def support_task(self, agent, origin, destination, answers):
        return Task(
            description=dedent(f"""
                I will provide you with an individual looking for guidance and advice on managing their emotions, stress, anxiety and other mental health issues arising due to Paris syndrome. 
                Use your knowledge of cognitive behavioral therapy, meditation techniques, mindfulness practices, and other therapeutic methods in order to create strategies that the individual can implement in order to improve their overall wellbeing.
                Use the {answers} to provide the best advice. 
                
                Origin: {origin}
                Destination: {destination}
                Answers: {answers}
                """),
            expected_output=dedent("""
                A summary of the conversation highlighting:
                1. Key concerns expressed by the user.
                2. Emotional reassurance provided.
                3. Recommended coping strategies.
                4. Also, recommend any mental helplines in the {destination} if you have the correct information. Otherwise, do not provide any fake numbers.
                """),
            agent=agent,
            
        )

    def cultural_task(self, agent, origin, destination, interests):
        return Task(
            description=dedent(f"""
                Educate the user about cultural differences and customs in {destination}.
                Provide tips on local etiquette and practices to help them adjust their expectations and also help address their problems.
                
                Origin: {origin}
                Destination: {destination}
                Interests: {interests}
                
                """),
            expected_output=dedent("""
                A cultural guide that includes:
                1. Key cultural differences to be aware of.
                2. Local etiquette and customs.
                3. Tips for blending in and avoiding cultural misunderstandings.
                4. Recommendations for culturally immersive experiences based on the user's interests.
                """),
            agent=agent
        )

