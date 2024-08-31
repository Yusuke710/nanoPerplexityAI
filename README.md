# nanoPerplexityAI

The simplest and most intuitive open-source implementation of an open source [perplexity.ai](https://www.perplexity.ai/), a large language model(LLM) service which cites information from Google. **No fancy GUI or LLM agents** are involved, just **200 lines of python code**. Please check out [this video](https://youtu.be/8zBDTnSYSoc) for more explanations!

Check out the [conversations](/example_outputs/) nanoPerplexityAI has generated 

![overview](/assets/example_response.png)

## Architecture

1. Get the user query
2. LLM checks the user query, decides whether to execute a Google search, and if searching, reformulates the user query into a Google-suited query to find relevant webpage URLs and fetch texts. (In practice, [PerplexityAI searches its already indexed sources](https://www.perplexity.ai/hub/faq/how-does-perplexity-work))
3. Build a prompt using `system prompt + webpage context + user query`
4. Call the LLM API to generate an answer
5. As LLM perform stream completion, save the LLM response into a markdown file for better visualization. 

#PerplexityAI does not reformat the search results and therefore not all search results are used and cited in the LLM response. This is because they prioritize displaying search results quickly and streaming LLM completion for a better user experience.

## Install
```
pip install googlesearch-python requests beautifulsoup4 lxml backoff openai 
```

## Quick Start
```
export OPENAI_API_KEY=<Your OpenAI API KEY>
python nanoPerplexityAI.py
```

The script will prompt you to type your question, then it will generate an answer in `playground.md`
You can type a key s for [s]ave and q for [q]uit

## View Real Time Generation
You can utilise Visual Studio Code to replicate the simpler version of PerplexityAI GUI. Open Preview of `playground.md` as you run `python nanoPerplexityAI.py` and you will see the real time generation!

### DEMO

![Gid](/assets/demo.gif)

Other ways involve opening in [Markdown Playground](https://dotmd-editor.vercel.app/) or pushing the output markdown files to your github repo for displaying markdown


## Acknowledgements
Thank you [perplexity.ai](https://www.perplexity.ai/) for the amazing idea and [clarity-ai](https://github.com/mckaywrigley/clarity-ai) and [Perplexica](https://github.com/ItzCrazyKns/Perplexica) for coding inspirations on the open-source implementation of perplexity.ai. 