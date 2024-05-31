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
from langchain_groq import ChatGroq
from langchain_community.llms import HuggingFaceEndpoint
from Home import google_api_key
llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True, 
                             temperature=0.2, google_api_key=google_api_key)

class MoodMender():

  def emotional_support(self):
    return Agent(
        role='Mental Health Advisor',
        goal='Provide mental health advice to the travelers suffering from Paris Syndrome',
        backstory="""A specialist who has knowledge of cognitive behavioral therapy, meditation techniques, mindfulness practices, and other therapeutic methods""",
        verbose=True,
        llm=llm,
        allow_delegation=False
    )

  def cultural_advisor(self):
    return Agent(
        role='Cultural Advisor',
        goal='Educate the user about cultural differences and help them adjust their expectations',
        backstory="""An expert in French culture and customs, dedicated to helping travelers understand and appreciate cultural differences to reduce cultural shock.""",
        verbose=True,
        llm=llm,
        allow_delegation=False
    )
