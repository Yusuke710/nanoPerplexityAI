## Speed Optimization

- Use a profiler to identify bottlenecks.
- Reduce response time:
  - Set a timeout when using `requests.get()`.
  - Note that [requests do not necessarily create a timeout](https://github.com/psf/requests/issues/3099). Instead, use [sys.trace to control the total timeout](https://stackoverflow.com/questions/21965484/timeout-for-python-requests-get-entire-response).
  - Use `lxml` instead of the HTML parser.

- Implement parallel processing for website parsing.

## Accuracy Optimization

- Utilize a rerank model to find most relevant sources to feed into LLM context. If the text exceeds your memory size, the rerank model runs inference multiple times, which can be time-consuming. Adjust parameters to avoid this.
- Prompt engineering
