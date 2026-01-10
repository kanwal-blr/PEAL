import streamlit as st
import os
import openai
# from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# LLM initialization (choose a model you have access to)
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    #model="gpt-4o",
    temperature=0.7,
)

# Prompt Template
evaluation_prompt = PromptTemplate.from_template("""
You are an expert English teacher. Your job is to evaluate the student's answer and give helpful feedback.

**Task:**
- Read the question and the student's answer.
- Decide if the answer is correct, partially correct, or incorrect.
- Give clear feedback on grammar, clarity, vocabulary, and missing information.
- Provide a suggested improved answer (only if needed).

**Output Format:**
Evaluation:
- Correctness: (Correct / Partially correct / Incorrect)
- Score: X/10
- Strengths: ...
- Areas to improve: ...
- Suggested improved answer: ...

**Here is the content to evaluate:**
Question: "{question}"
Student Answer: "{answer}"
""")

# 1. Define the prompt template
peel_prompt_string = """
You are an experienced IGCSE examiner. Your task is to evaluate a middle school student’s written response using the PEEL structure: Point, Evidence, Explanation, Link.

Evaluate the answer against these criteria:

1. POINT — Is the argument clearly stated and directly answering the question?
2. EVIDENCE — Is there relevant, accurate, and specific supporting evidence?
3. EXPLANATION — Does the student explain how the evidence supports the point, showing understanding and analysis?
4. LINK — Does the student connect back to the question or provide a clear transition?

Expectation form the Student's Answer:
There should be one introductory paragraph followed by three body paragraphs and one conclusion paragraph. 
The introductory paragraph should first begin with the title of the story, 
then the name of the author,
then a short one or two lines about the content of the extract or the content of the scene, 
followed by a thesis statement. 
The thesis statement should provide a clear answer to the question and outline all the points which will be highlighted in the following essay. 
The body paragraphs should follow the PEEL format, but language aspects also require to be mentioned. 
In language aspects, you must identify a literary device or other forms of language used by the author to bring out what they're trying to say. 
You can either use the PETAL (point, Evidence, Explanation, Link) format for this or provide a separate body paragraph for the same. 
If you're providing a separate body paragraph, you have two paragraphs which explain points and one paragraph which highlights the literary devices. 
Then in the conclusion, you must summarize all the points and restate the thesis statement. 
Throughout the essay, you should not narrate and you need to be specific to the answer. 
However, evidence must be explained and some content and background may be provided by doing the same.

Evaluation Criteria:
- 10 marks for the content of the answer
- 5 marks for quality of writing (grammar, vocabulary, clarity)
- Total: 15 marks

Provide your feedback in paragraph format using the structure below:
- Start with giving a score out of 15 based on overall effectiveness. (Output this in bold and add newline after this line).
- Add a paragrah for each of the sections below and seperate them by a newline.
- Provide a judgment of how well the PEEL structure is followed.
- Comment specifically on the strengths in Point, Evidence, Explanation, and Link.
- Provide 2–3 clear suggestions for improvement (EBI: Even Better If…).
- Conclude with a brief summary sentence encouraging improvement.

Tone: Constructive, supportive, and academically appropriate for IGCSE level.


QUESTION:
{question}

STUDENT_ANSWER:
{student_answer}
"""

peel_prompt = PromptTemplate(
    template=peel_prompt_string,
    input_variables=["question", "student_answer"],
)


def evaluate_answer(question_text, answer_text):
    prompt_value = peel_prompt.format(
        question=question_text,
        student_answer=answer_text
    )
    response = llm.invoke(prompt_value)
    return response.content

# Streamlit App
st.title("PEEL Evaluator")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Enter PEEL Question ")
    # question_file = st.file_uploader("", type=["csv",".txt"], key="question")
    # if question_file is not None:
    #     st.info(f"Question file: {question_file.name}")
    question_text = st.text_area("Paste or type the question here", height=500, key="question_text")

with col2:
    st.subheader("Enter PEEL Answer")
    # answer_file = st.file_uploader("", type=["csv",".txt"], key="answer")
    # if answer_file is not None:
    #     st.info(f"Answer file: {answer_file.name}")
    answer_text = st.text_area("Paste or type the answer here", height=500, key="answer_text")

# Add custom button styling
st.markdown("""
    <style>
    div.stButton > button {
        background-color: #2E86AB;
        color: white;
        font-size: 20px;
        font-weight: bold;
        padding: 14px 40px;
        border-radius: 8px;
        border: none;
        width: 100%;
        margin: 12px 0px;
    }
    div.stButton > button:hover {
        background-color: #1a6682;
        border: none;
    }
    </style>    
""", unsafe_allow_html=True)



# Center the button using columns
col1, col2, col3 = st.columns([1,2,1])
with col2:
    evaluate = st.button("Evaluate")

if evaluate:
    if question_text is None or answer_text is None:
        st.error("Please enter both the question and answer before evaluating.")
    else:
        with st.spinner("Evaluating..."):
            evaluation_result = evaluate_answer(question_text, answer_text)
        st.success("Evaluation Complete!")
        st.markdown("### Evaluation Result:")
        st.text_area("", value=evaluation_result, height=500)
    # else:
    #     # Add horizontal line separator
    #     st.markdown("<hr style='border:0;border-top:2px solid #eee;margin-top:18px;margin-bottom:18px;'/>", 
    #             unsafe_allow_html=True)

    #     # Add evaluation metrics header
    #     st.markdown("<h2 style='color:#2E86AB;margin:0 0 8px 0;font-weight:800;'>Evaluation Metrics</h2>", 
    #             unsafe_allow_html=True)

    #     # Display score
    #     st.markdown("<div style='font-size:20px;font-weight:700;margin-bottom:6px;'>" 
    #             "Your Score: <span style='font-size:26px;color:#D9534F;'>6/10</span></div>", 
    #             unsafe_allow_html=True)

    #     # Display comments section
    #     st.markdown("""
    #         <div style='background:#FFF3CD;border-left:6px solid #FFE8A1;padding:12px;border-radius:6px;'>
    #             <strong style='color:#856404;'>Comments:</strong>
    #             <span style='margin-left:8px;color:#6B4F00;font-weight:600;'>PEAL Better Next Time!</span>
    #         </div>
    #         """, 
    #         unsafe_allow_html=True)
