import os
from dotenv import load_dotenv

#from openai import OpenAI
import openai
import re,time
from firecrawl import FirecrawlApp
import json

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')

load_dotenv()