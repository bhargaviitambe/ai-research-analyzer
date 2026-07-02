import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)


def ask_gemini(prompt: str) -> str:
    """
    Sends a prompt to Gemini and returns the response as a string.
    This is the core function everything else uses.
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


def summarize_paper(full_text: str) -> dict:
    """
    Generates section-wise summaries of the research paper.
    """
    prompt = f"""
    You are an expert research assistant. Analyze the following research paper 
    and provide structured summaries for each section.

    Paper text:
    {full_text[:8000]}

    Provide summaries in this exact format:
    ABSTRACT SUMMARY: (2-3 sentences summarizing the abstract)
    METHODOLOGY SUMMARY: (2-3 sentences summarizing the methodology)
    RESULTS SUMMARY: (2-3 sentences summarizing the results)
    CONCLUSION SUMMARY: (2-3 sentences summarizing the conclusion)
    """
    response = ask_gemini(prompt)
    return parse_summary_response(response)


def parse_summary_response(response: str) -> dict:
    """
    Parses Gemini's structured text response into a Python dictionary.
    """
    result = {
        "abstract_summary": "",
        "methodology_summary": "",
        "results_summary": "",
        "conclusion_summary": ""
    }

    lines = response.split("\n")
    for line in lines:
        if line.startswith("ABSTRACT SUMMARY:"):
            result["abstract_summary"] = line.replace("ABSTRACT SUMMARY:", "").strip()
        elif line.startswith("METHODOLOGY SUMMARY:"):
            result["methodology_summary"] = line.replace("METHODOLOGY SUMMARY:", "").strip()
        elif line.startswith("RESULTS SUMMARY:"):
            result["results_summary"] = line.replace("RESULTS SUMMARY:", "").strip()
        elif line.startswith("CONCLUSION SUMMARY:"):
            result["conclusion_summary"] = line.replace("CONCLUSION SUMMARY:", "").strip()

    return result


def detect_research_gaps(full_text: str) -> list:
    """
    Identifies potential research gaps and limitations in the paper.
    """
    prompt = f"""
    You are an expert research assistant. Read the following research paper 
    and identify research gaps, limitations, and areas that need more investigation.

    Paper text:
    {full_text[:8000]}

    List exactly 5 research gaps in this format:
    1. (gap)
    2. (gap)
    3. (gap)
    4. (gap)
    5. (gap)

    Only output the numbered list, nothing else.
    """
    response = ask_gemini(prompt)
    return parse_numbered_list(response)


def suggest_future_work(full_text: str) -> list:
    """
    Suggests future research directions based on the paper.
    """
    prompt = f"""
    You are an expert research assistant. Based on the following research paper,
    suggest future research directions that would advance this field.

    Paper text:
    {full_text[:8000]}

    List exactly 5 future work suggestions in this format:
    1. (suggestion)
    2. (suggestion)
    3. (suggestion)
    4. (suggestion)
    5. (suggestion)

    Only output the numbered list, nothing else.
    """
    response = ask_gemini(prompt)
    return parse_numbered_list(response)


def extract_citations(full_text: str) -> dict:
    """
    Extracts paper metadata and generates citations in multiple formats.
    """
    prompt = f"""
    You are an expert research assistant. Extract the citation information 
    from the following research paper and format it in multiple citation styles.

    Paper text:
    {full_text[:3000]}

    Provide the output in this exact format:
    TITLE: (paper title)
    AUTHORS: (comma separated list of authors)
    YEAR: (publication year)
    APA: (full APA citation)
    MLA: (full MLA citation)
    IEEE: (full IEEE citation)
    """
    response = ask_gemini(prompt)
    return parse_citation_response(response)


def parse_citation_response(response: str) -> dict:
    """
    Parses Gemini's citation response into a dictionary.
    """
    result = {
        "title": "",
        "authors": [],
        "year": "",
        "apa": "",
        "mla": "",
        "ieee": ""
    }

    lines = response.split("\n")
    for line in lines:
        if line.startswith("TITLE:"):
            result["title"] = line.replace("TITLE:", "").strip()
        elif line.startswith("AUTHORS:"):
            authors_str = line.replace("AUTHORS:", "").strip()
            result["authors"] = [a.strip() for a in authors_str.split(",")]
        elif line.startswith("YEAR:"):
            result["year"] = line.replace("YEAR:", "").strip()
        elif line.startswith("APA:"):
            result["apa"] = line.replace("APA:", "").strip()
        elif line.startswith("MLA:"):
            result["mla"] = line.replace("MLA:", "").strip()
        elif line.startswith("IEEE:"):
            result["ieee"] = line.replace("IEEE:", "").strip()

    return result


def answer_question(context_chunks: list, question: str) -> str:
    """
    Answers a user's question based only on the relevant chunks from the paper.
    This is the RAG answer generation step.
    """
    context = "\n\n".join(context_chunks)

    prompt = f"""
    You are an expert research assistant. Answer the following question 
    based ONLY on the provided context from a research paper.
    If the answer is not in the context, say "This information is not found in the paper."

    Context from paper:
    {context}

    Question: {question}

    Answer:
    """
    return ask_gemini(prompt)


def extract_keywords(full_text: str) -> list:
    """
    Extracts the main topic and keywords from the paper.
    Used by the similar papers feature to search for related work.
    """
    prompt = f"""
    You are an expert research assistant. Extract the main research topic 
    and 5 key terms from the following research paper.

    Paper text:
    {full_text[:3000]}

    Respond in this exact format:
    TOPIC: (one sentence describing the main topic)
    KEYWORDS: (exactly 5 keywords separated by commas)
    """
    response = ask_gemini(prompt)
    return parse_keywords_response(response)


def parse_keywords_response(response: str) -> dict:
    """
    Parses Gemini's keyword response into a dictionary.
    """
    result = {"topic": "", "keywords": []}

    lines = response.split("\n")
    for line in lines:
        if line.startswith("TOPIC:"):
            result["topic"] = line.replace("TOPIC:", "").strip()
        elif line.startswith("KEYWORDS:"):
            keywords_str = line.replace("KEYWORDS:", "").strip()
            result["keywords"] = [k.strip() for k in keywords_str.split(",")]

    return result


def parse_numbered_list(response: str) -> list:
    """
    Parses a numbered list response from Gemini into a Python list.
    Works for research gaps and future work responses.
    """
    items = []
    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            # Remove the number and period at the start (e.g. "1. ")
            item = line.split(".", 1)[1].strip()
            if item:
                items.append(item)
    return items