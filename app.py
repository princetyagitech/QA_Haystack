import streamlit as st
import requests
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http
import os
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from pprint import pprint


document_store = InMemoryDocumentStore(use_bm25=True)

doc_dir = "data/build_your_first_question_answering_system"

fetch_archive_from_http(
    url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt1.zip",
    output_dir=doc_dir
)
files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline = TextIndexingPipeline(document_store)
indexing_pipeline.run_batch(file_paths=files_to_index)

retriever = BM25Retriever(document_store=document_store)


reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)


pipe = ExtractiveQAPipeline(reader, retriever)

st.title("Question Answering system")
question = st.text_area("Enter your question:") # taking query
if st.button("Answer"):

    answers=pipe.run(query=question,params={
        "Retriever": {"top_k": 10},
        "Reader": {"top_k": 5}
    })
    # show results
    if answers:
        
        st.write(pprint(answers))
        
        st.write('---')
   


