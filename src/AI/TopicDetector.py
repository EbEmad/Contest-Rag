"""
AI Topic Detector
Automatically detects which curriculum topics a book covers.
"""

from typing import List
from motor.motor_asyncio import AsyncIOMotorDatabase
from AI.llm.LLMInterface import LLMInterface
from bson.objectid import ObjectId
import logging

logger = logging.getLogger(__name__)

TOPIC_DETECTION_PROMPT = """
You are an educational content analyzer.
Available topics in our curriculum:
{available_topics}
Book content sample:
---
{book_sample}
---
Task: Identify which topics from our curriculum this book covers.
Rules:
1. Only return topics that are EXPLICITLY covered in the sample
2. Return topic names EXACTLY as they appear in the available topics list
3. Return as a JSON array of topic names
Example response:
["Linear Equations", "Quadratic Equations"]
Your response (JSON array only):
"""

