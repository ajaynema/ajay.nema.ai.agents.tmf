from crewai_tools import ScrapeWebsiteTool, FileWriterTool, TXTSearchTool
import requests
from dotenv import load_dotenv
load_dotenv()

# Initialize the tool, potentially passing the session
tool = ScrapeWebsiteTool(website_url='https://www.telstra.com.au')  

# Extract the text
text = tool.run()
print(text)

file_writer_tool = FileWriterTool()

# Write content to a file in a specified directory
result = file_writer_tool._run(filename='telstra.txt', content = text, directory = '', overwrite="True")
print(result)

import os
from crewai_tools import TXTSearchTool

# Initialize the tool with a specific text file, so the agent can search within the given text file's content
tool = TXTSearchTool(txt='telstra.txt')

from crewai import Agent, Task, Crew

context = tool.run('What is telstra vision?')

while True :
    query = input("Ask me question (exit for quit) : ")
    if query  == "exit":
       break
    data_analyst = Agent(
        role='Educator',
        goal=f'Based on the context provided, answer the question -'+query+' ? Context - {context}',
        backstory='You are a data expert',
        verbose=True,
        allow_delegation=False,
        tools=[tool]
    )

    review_analyst = Agent(
        role='Reviewer',
        goal=f'review the response if not correct and satisfactory correct the response -'+query+' ? Context - {context} . Get the response review by the reviewer agent',
        backstory='You are a expert reviewer',
        verbose=True,
        allow_delegation=False,
        tools=[tool]
    )

    test_task = Task(
        description="Understand the topic and give the correct response",
        tools=[tool],
        agent=data_analyst,
        expected_output='Give a correct response'
    )

    review_task = Task(
        description="Understand the topic and review the response",
        tools=[tool],
        agent=review_analyst,
        expected_output='Review and correct the response'
    )

    crew = Crew(
        agents=[data_analyst,review_analyst],
        tasks=[test_task,review_task]
    )

    output = crew.kickoff()