# langchain-weaviate-rag

Just a simple bit of Python code that uses Lanchain and the Weaviate client to extract text
from a blog post, load it into a local Weaviate DB, and prompt an OpenAI LLM to summarize the
contents of the post.

## Prerequisites

You need to run the Weaviate DB as a docker container before running the app.
https://weaviate.io/developers/weaviate/installation/docker-compose#default-weaviate-environment

You also need an OpenAI API key exported as OPENAI_API_KEY in your environment.

## Running

`python main.py`
