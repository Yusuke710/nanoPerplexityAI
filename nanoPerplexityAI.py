import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from googlesearch import search
import requests
from bs4 import BeautifulSoup
import backoff
import openai

# -----------------------------------------------------------------------------
# Default configuration
NUM_SEARCH = 10  # Number of links to parse from Google
SEARCH_TIME_LIMIT = 3  # Max seconds to request website sources before skipping to the next URL
TOTAL_TIMEOUT = 6  # Overall timeout for all operations
MAX_CONTENT = 500  # Number of words to add to LLM context for each search result
MAX_TOKENS = 1000 # Maximum number of tokens LLM generates
LLM_MODEL = 'gpt-4o-mini' #'gpt-3.5-turbo' #'gpt-4o'
system_prompt_search = """You are a helpful assistant whose primary goal is to decide if a user's query requires a Google search."""
search_prompt = """
Decide if a user's query requires a Google search. You should use Google search for most queries to find the most accurate and updated information. Follow these conditions:

- If the query does not require Google search, you must output "ns", short for no search.
- If the query requires Google search, you must respond with a reformulated user query for Google search.
- User query may sometimes refer to previous messages. Make sure your Google search considers the entire message history.

User Query:
{query}
"""
system_prompt_answer = """You are a helpful assistant who is expert at answering user's queries"""
answer_prompt = """Generate a response that is informative and relevant to the user's query
User Query:
{query}
"""
system_prompt_cited_answer = """You are a helpful assistant who is expert at answering user's queries based on the cited context."""
cited_answer_prompt = """
Provide a relevant, informative response to the user's query using the given context (search results with [citation number](website link) and brief descriptions).

- Answer directly without referring the user to any external links.
- Use an unbiased, journalistic tone and avoid repeating text.
- Format your response in markdown with bullet points for clarity.
- Cite all information using [citation number](website link) notation, matching each part of your answer to its source.

Context Block:
{context_block}

User Query:
{query}
"""
# -----------------------------------------------------------------------------

# Set up OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

client =openai.OpenAI(api_key=OPENAI_API_KEY)

def get_query():
    """Prompt the user to enter a query."""
    return input("Enter your question: ")

def trace_function_factory(start):
    """Create a trace function to timeout request"""
    def trace_function(frame, event, arg):
        if time.time() - start > TOTAL_TIMEOUT:
            raise TimeoutError('Website fetching timed out')
        return trace_function
    return trace_function

def fetch_webpage(url, timeout):
    """Fetch the content of a webpage given a URL and a timeout."""
    start = time.time()
    sys.settrace(trace_function_factory(start))
    try:
        print(f"Fetching link: {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        paragraphs = soup.find_all('p')
        page_text = ' '.join([para.get_text() for para in paragraphs])
        return url, page_text
    except (requests.exceptions.RequestException, TimeoutError) as e:
        print(f"Error fetching {url}: {e}")
    finally:
        sys.settrace(None)
    return url, None

def parse_google_results(query, num_search=NUM_SEARCH, search_time_limit=SEARCH_TIME_LIMIT):
    """Perform a Google search and parse the content of the top results."""
    urls = search(query, num_results=num_search)
    max_workers = os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() returns None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_webpage, url, search_time_limit): url for url in urls}
        return {url: page_text for future in as_completed(future_to_url) if (url := future.result()[0]) and (page_text := future.result()[1])}

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def llm_check_search(query, msg_history=None, llm_model=LLM_MODEL):
    """Check if query requires search and execute Google search."""
    prompt = search_prompt.format(query=query)
    msg_history = msg_history or []
    new_msg_history = msg_history + [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "system", "content": system_prompt_search}, *new_msg_history],
        max_tokens=30
    ).choices[0].message.content

    # check if the response contains "ns"
    cleaned_response = response.lower().strip()
    if re.fullmatch(r"\bns\b", cleaned_response):
        print("No Google search required.")
        return None
    else:
        print(f"Performing Google search: {cleaned_response}")
        search_dic = parse_google_results(cleaned_response)
        return search_dic

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def llm_answer(query, msg_history=None, search_dic=None, llm_model=LLM_MODEL, max_content=MAX_CONTENT, max_tokens=MAX_TOKENS):
    """Build the prompt for the language model including the search results context."""
    if search_dic:
        context_block = "\n".join([f"[{i+1}]({url}): {content[:max_content]}" for i, (url, content) in enumerate(search_dic.items())])
        prompt = cited_answer_prompt.format(context_block=context_block, query=query)
        system_prompt = system_prompt_cited_answer
    else:
        prompt = answer_prompt.format(query=query)
        system_prompt = system_prompt_answer

    """Generate a response using the OpenAI language model."""
    msg_history = msg_history or []
    new_msg_history = msg_history + [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "system", "content": system_prompt}, *new_msg_history],
        max_tokens=max_tokens
    ).choices[0].message.content

    new_msg_history.append({"role": "assistant", "content": response})
    return response, new_msg_history

def renumber_citations(response):
    """Renumber citations in the response to be sequential."""
    # Find all unique citations in the order of their first appearance
    citations = list(dict.fromkeys(map(int, re.findall(r'\[\(?(\d+)\)?\]', response))))

    citation_map = {old: new for new, old in enumerate(citations, 1)}
    for old, new in citation_map.items():
        response = re.sub(rf'\[\(?{old}\)?\]', f'[{new}]', response)
    return response, citation_map

def generate_citation_links(citation_map, search_dic):
    """Generate citation links based on the renumbered response."""
    cited_links = [f"{new}. {list(search_dic.keys())[old-1]}" for old, new in citation_map.items()]
    return "\n".join(cited_links)

def save_markdown(query, response, search_dic, file_path):
    """Renumber citations, then save the query, response, and sources to a markdown file."""
    links_block = ""
    if search_dic:
        response, citation_map = renumber_citations(response)
        links_block = generate_citation_links(citation_map, search_dic) if citation_map else ""
    output_content = f"# {query}\n\n## Sources\n{links_block}\n\n## Answer\n{response}\n\n" if search_dic else f"# {query}\n\n## Answer\n{response}\n\n"

    # Open the file with the specified mode ('w' for write or 'a' for append)
    with open(file_path, 'a') as file:
        file.write(output_content)
    print(f"AI response saved into {file_path}")

def main():
    """Main function to execute the search, generate response, and save to markdown."""
    msg_history = None
    file_path = None

    while True:
        query = get_query()
        file_path = file_path or f"{query}.md"

        search_dic = llm_check_search(query, msg_history)
        response, msg_history = llm_answer(query, msg_history, search_dic)
        save_markdown(query, response, search_dic, file_path)

        print("-" * 40)
        print("Press Ctrl+C to quit")

if __name__ == "__main__":
    main()
