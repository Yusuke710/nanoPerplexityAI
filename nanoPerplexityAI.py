import os
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import markdown
from googlesearch import search

# -----------------------------------------------------------------------------
# default config
max_links = 5 # number of links to parse from google
max_content = 300 # number of words to add to LLM context for each search result
search_time_limit = 3 # max seconds to request website sources until you skip to the next url
model = 'gpt-4o' # 'gpt-3.5-turbo' 
output_md = 'response.md' 
# -----------------------------------------------------------------------------

# Set up OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key is None:
    raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=openai_api_key)

def get_query():
    query = input("Enter your query: ")
    return query
    
def google_parse_webpages(query, max_links):
    search_dic = {}
    for url in search(query, num_results=max_links):
        try:
            print(f"Fetching link: {url}")
            response = requests.get(url, timeout=search_time_limit)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            page_text = ' '.join([para.get_text() for para in paragraphs])
            search_dic[url] = page_text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            continue  # Skip the failed link
    return search_dic

def build_prompt(query, search_dic):
    # search_dic contains a dictionary with links as keys and contents as values
    # context block comply the reference format e.g. text related to the search result ([1](https://www.example.com))
    context_list = [f"[{i+1}]({url}): {content[:max_content]}" for i, (url, content) in enumerate(search_dic.items())]
    context_block = "\n".join(context_list)

    # This is where magic happens
    system_message = f"""You are an AI model who is expert at answering user's queries based on the cited context.

    Generate a response that is informative and relevant to the user's query based on provided context (the context consists of search results containing a key with [citation number](website link) and brief description of the content of that page).
    You must use this context to answer the user's query in the best way possible. Use an unbiased and journalistic tone in your response. Do not repeat the text.
    You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself.
    Your responses should be medium to long in length, be informative and relevant to the user's query. You must use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
    You have to cite the answer using [citation number](website link) notation. You must cite the sentences with their relevant context number. You must cite each and every part of the answer so the user can know where the information is coming from.
    Anything inside the following context block provided below is for your knowledge returned by the search engine and is not shared by the user. You have to answer questions on the basis of it and cite the relevant information from it but you do not have to 
    talk about the context in your response.
    context block:
    {context_block}
    """
    return [{"role": "system", "content": system_message}, {"role": "user", "content": query}]

def llm_openAI(prompt, model):
    response = client.chat.completions.create(
        model=model,
        messages=prompt
    )
    return response.choices[0].message.content

def save_markdown(query, response, search_dic, output_md=output_md):
    # Format sources as follow
    # Sources:
    # 1. https://www.exampleA.com  
    # 2. https://www.exampleB.com 
    links_list = [f"{i+1}. {url}" for i, (url, _) in enumerate(search_dic.items())]
    links_block = "\n".join(links_list)

    output = f"# Query:\n{query}\n\n# Response:\n{response}\n\n# Search Results:\n{links_block}"
    with open(output_md, "w") as file:
        file.write(output)

def main():
    query = get_query() # query user for input
    search_dic = google_parse_webpages(query, max_links) # search google and scrape website info
    prompt = build_prompt(query, search_dic) # combine system prompt, user prompt and website info
    response = llm_openAI(prompt, model) # inference using LLM 
    save_markdown(query, response, search_dic) # save the result into markdown for visualisation

if __name__ == "__main__":
    main()
