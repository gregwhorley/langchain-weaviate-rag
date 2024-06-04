import bs4
import weaviate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


if __name__ == "__main__":
    loader = WebBaseLoader(
        web_paths=(
            "https://weaviate.io/blog/why-is-vector-search-so-fast",
        ),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_="markdown"
            )
        )
    )
    docs = loader.load()

    text_blob = ""
    metadata_list = []
    for item in docs:
        text_blob += item.page_content
        metadata_list.append(item.metadata)
    documents = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0
    ).split_text(text_blob)

    embeddings = OpenAIEmbeddings()
    weaviate_client = weaviate.connect_to_local()

    docsearch = WeaviateVectorStore.from_texts(
        documents,
        embeddings,
        client=weaviate_client,
        metadatas=metadata_list,
    )
    retriever = docsearch.as_retriever()
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    output = rag_chain.invoke("Summarize the performance characteristics of vector search")
    print(output)
