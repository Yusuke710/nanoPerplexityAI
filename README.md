# nanoPerplexityAI
![meme](/assets/meme.png)

The simplest and most intuitive open-source implementation of an open source [perplexity.ai](https://www.perplexity.ai/), a large language model(LLM) service which cites information from Google. No fancy GUI or LLM agents are involved, just **100 lines of python code**.
Please check out [this video(to be uploaded soon)]() for more explanations!

## Architecture

1. Get the user query
2. Search Google to find relevant webpage URLs and fetch texts
3. Use [rerank model(cross encoder)](https://www.sbert.net/examples/applications/retrieve_rerank/README.html) to find relevance scores for each search results
4. Build a prompt using `system prompt + webpage context + user query`
5. Call the LLM API to generate a response
6. Format citations and save the LLM response into a markdown file to visualize it

## Install
```
pip install googlesearch-python requests beautifulsoup4 sentence-transformers openai 
```

## Quick Start
```
export OPENAI_API_KEY=<Your OpenAI API KEY>
python nanoPerplexityAI.py
```

The script will prompt you to type your question, then it will generate a response in `response.md`

## View the LLM Responses:
There are several ways to visualize the responses easily:
- Open in your editor, e.g., VScode
- Open in [Markdown Playground](https://dotmd-editor.vercel.app/)
- Push them to your github repo

Check out the [responses](/example_outputs/) nanoPerplexityAI has already generated 

![Response](/assets/example_response.png)


## Acknowledgements
Thank you [perplexity.ai](https://www.perplexity.ai/) for the amazing idea and [clarity-ai](https://github.com/mckaywrigley/clarity-ai) and [Perplexica](https://github.com/ItzCrazyKns/Perplexica) for coding inspirations on the open-source implementation of perplexity.ai. 