"""
This script summarizes a user-provided text and then allows the user to ask questions about the summary.
It uses HuggingFace transformers and LangChain integrations.
"""

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers.utils.logging import set_verbosity_error

# Disable transformer logging for cleaner output
set_verbosity_error()

# Create a HuggingFace summarization pipeline using the BART model
summarization_pipline = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
summarizer = HuggingFacePipeline(pipeline=summarization_pipline)

# Create another summarization pipeline for additional refinement
refinement_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
refiner = HuggingFacePipeline(pipeline=refinement_pipeline)

# Create a question-answering pipeline using a RoBERTa model trained on SQuAD2.0
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0)

# Define a prompt template that inserts user input text and desired summary length
summary_template = PromptTemplate.from_template(
    "Summarize the following text in a {length} way \n\n{text}"
)

# Create a chain: prompt -> summarizer -> refiner
summarization_chain = summary_template | summarizer | refiner

# Get input text and summary length from the user
text_to_summarize = input("\nEnter text to summarize:\n")
length = input("\nEnter the length (short/medium/long): ")

# Generate a summary using the chain
summary = summarization_chain.invoke({"text": text_to_summarize, "length": length})

# Display the generated summary
print("\n **Generated summary:**")
print(summary)

# Loop for asking questions about the summary
while True:
    question = input("\nAsk a question about the summary (or type 'exit' to stop):\n")

    if question.lower() == "exit":
        break

    if not question.strip():
        print("⚠️ You entered an empty question. Please try again.")
        continue

    # Run the QA pipeline on the summary with the given question
    qa_result = qa_pipeline(question=question, context=summary)

    # Print the answer to the question
    print("\n **Answer:**")
    print(qa_result["answer"])
