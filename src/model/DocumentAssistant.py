import json
import os
from os.path import join
from tqdm.auto import tqdm
from typing import List, Optional, Literal
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
import torch
import chromadb
from chromadb import PersistentClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import docx
from pathlib import Path


class DocumentAssistant:
    def __init__(self, base_model_id="microsoft/Phi-4-mini-instruct"):
        self.base_model_id = base_model_id
        self.device = "cuda:0"
        self.torch_dtype = None
        
        # Initialize models and tokenizers
        self.llm_model, self.llm_tokenizer = self._load_llm_model()
        self.embedding_model, self.embedding_tokenizer = self._load_embedding_model()
        
        # Initialize ChromaDB client
        self.client = PersistentClient(path="chroma-db")
        
        # Dictionary to keep track of collections
        self.collections = {}

    def _load_llm_model(self):
        model=AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            device_map="auto",
            torch_dtype=self.torch_dtype
        )

        tokenizer=AutoTokenizer.from_pretrained(self.base_model_id)
        return model,tokenizer
    
    def _load_embedding_model(self):
        model_name = "intfloat/multilingual-e5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return model, tokenizer
    
    def embedd_text(self,text):
        inputs=self.embedding_tokenizer(text,"")