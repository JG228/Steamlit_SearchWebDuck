from googlesearch import search
from langchain.tools.base import BaseTool

class GoogleSearchRun(BaseTool):
    def __init__(self, name="GoogleSearch"):
        super().__init__(name)

    def run(self, query, num_results=10):
        search_results = []
        for result in search(query, num_results=num_results):
            search_results.append(result)
        return search_results
