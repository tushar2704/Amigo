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