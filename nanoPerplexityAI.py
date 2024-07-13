import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from googlesearch import search
import requests
from bs4 import BeautifulSoup
from sentence_transformers import CrossEncoder
from openai import OpenAI
import markdown

# -----------------------------------------------------------------------------
# Default configuration
NUM_SEARCH = 20  # Number of links to parse from Google
SEARCH_TIME_LIMIT = 3  # Max seconds to request website sources before skipping to the next URL
MAX_CONTENT = 400  # Number of words to add to LLM context for each search result
RERANK_TOP_K = 5 # Top k ranked search results going into context of LLM
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-12-v2'  # Max tokens = 512
LLM_MODEL = 'gpt-3.5-turbo'  # 'gpt-4o' 
OUTPUT_MD = 'response.md'
# -----------------------------------------------------------------------------

# Set up OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if OPENAI_API_KEY is None:
    raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)

def get_query():
    """Prompt the user to enter a query."""
    return input("Enter your query: ")

def fetch_webpage(url, timeout):
    """Fetch the content of a webpage given a URL and a timeout."""
    try:
        print(f"Fetching link: {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        page_text = ' '.join(para.get_text() for para in paragraphs)
        return url, page_text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return url, None

def google_parse_webpages(query, num_search=NUM_SEARCH, search_time_limit=SEARCH_TIME_LIMIT):
    """Perform a Google search and parse the content of the top results."""
    urls = search(query, num_results=num_search)
    with ThreadPoolExecutor() as executor:
        future_to_url = {executor.submit(fetch_webpage, url, search_time_limit): url for url in urls}
        return {url: page_text for future in as_completed(future_to_url) if (url := future.result()[0]) and (page_text := future.result()[1])}

def rerank_search_results(query, search_dic, rerank_model=RERANK_MODEL, rerank_top_k=RERANK_TOP_K):
    """Rerank search results based on relevance to the query using a CrossEncoder model."""
    model = CrossEncoder(rerank_model)
    query_context_pairs = [(query, content) for content in search_dic.values()]
    scores = model.predict(query_context_pairs)
    top_results = sorted(zip(search_dic.keys(), search_dic.values(), scores), key=lambda x: x[2], reverse=True)[:rerank_top_k]
    return {link: content for link, content, _ in top_results}

def build_prompt(query, search_dic, max_content=MAX_CONTENT):
    """Build the prompt for the language model including the search results context."""
    context_list = [f"[{i+1}]({url}): {content[:max_content]}" for i, (url, content) in enumerate(search_dic.items())]
    context_block = "\n".join(context_list)
    system_message = f"""
    You are an AI model who is expert at answering user's queries based on the cited context.

    Generate a response that is informative and relevant to the user's query based on provided context (the context consists of search results containing a key with [citation number](website link) and brief description of the content of that page).
    You must use this context to answer the user's query in the best way possible. Use an unbiased and journalistic tone in your response. Do not repeat the text.
    You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself.
    Your responses should be medium to long in length, be informative and relevant to the user's query. You must use markdown to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
    You have to cite the answer using [citation number](website link) notation. You must cite the sentences with their relevant context number. You must cite each and every part of the answer so the user can know where the information is coming from.
    Anything inside the following context block provided below is for your knowledge returned by the search engine and is not shared by the user. You have to answer questions on the basis of it and cite the relevant information from it but you do not have to 
    talk about the context in your response.
    context block:
    {context_block}
    """
    return [{"role": "system", "content": system_message}, {"role": "user", "content": query}]

def llm_openai(prompt, llm_model=LLM_MODEL):
    """Generate a response using the OpenAI language model."""
    response = client.chat.completions.create(
        model=llm_model,
        messages=prompt
    )
    return response.choices[0].message.content

def save_markdown(query, response, search_dic, output_md=OUTPUT_MD):
    """Save the query, response, and sources to a markdown file."""
    cited_numbers = set(map(int, re.findall(r'\[(\d+)\]', response)))
    cited_links = [f"{i+1}. {url}" for i, (url, _) in enumerate(search_dic.items()) if (i + 1) in cited_numbers]
    links_block = "\n".join(cited_links)
    output = f"# Query:\n{query}\n\n# Response:\n{response}\n\n# Sources:\n{links_block}"
    with open(output_md, "w") as file:
        file.write(output)

def main():
    """Main function to execute the search, rerank results, generate response, and save to markdown."""
    query = get_query() 
    search_dic = google_parse_webpages(query)
    reranked_search_dic = rerank_search_results(query, search_dic)
    prompt = build_prompt(query, reranked_search_dic)
    response = llm_openai(prompt)
    save_markdown(query, response, reranked_search_dic)

if __name__ == "__main__":
    main()
