## Speed Optimization
- Use a profiler to identify bottlenecks.
- Reduce response time:
  - Set a timeout when using `requests.get()`.
  - Note that [requests do not necessarily create a timeout](https://github.com/psf/requests/issues/3099). Instead, use [sys.trace to control the total timeout](https://stackoverflow.com/questions/21965484/timeout-for-python-requests-get-entire-response).
  - Use `lxml` instead of the HTML parser.

- Implement parallel processing for website parsing.

## Accuracy Optimization
- This application heavily relies on the performance of Google Search:
  - Relevancy of websites to user queries [TF-IDF](https://youtu.be/zLMEnNbdh4Q?si=WBZCkwryzOrkhfkX)
  - Ranking high quality websites higher in search results [Page Rank](https://youtu.be/JGQe4kiPnrU?si=mJkXOL2o5lDdxGon)
- Improve prompt engineering to ensure LLM answers are contextually accurate and follow citation formats.
