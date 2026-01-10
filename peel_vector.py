import os
import openai
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# -----------------------
#  CONFIG / SECRETS
# -----------------------
# On Streamlit Cloud, set this in:
# Settings -> Secrets -> add:
# OPENAI_API_KEY = "sk-..."
import os
#os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
# -----------------------
#  EXAMPLE EVALUATIONS
# -----------------------
# TODO: Replace the placeholder texts with your REAL marked examples.
EXAMPLE_EVALUATIONS = [
    {
        "label": "high_band_example",
        "text": """EXAMPLE
Question: How does the writer create tension in this passage?
Student answer:
[Put the full strong student answer here.]

Teacher feedback:
**Score: 14/15**
This response shows a strong understanding of the passage and follows the required essay structure effectively. The introduction clearly states the title, author, and context of the extract, and the thesis statement directly answers the question by outlining the key ways tension is created. The body paragraphs generally follow a PEEL structure, and the student supports their points with relevant quotations and clear explanations. Language aspects such as sensory imagery and short, abrupt sentences are identified and analysed, which strengthens the response. Minor improvements in depth of analysis could push this into full marks.
"""
    },
    {
        "label": "mid_band_example",
        "text": """EXAMPLE
Question: In what ways is the character shown as selfish?
Student answer:
[Put a mid-level student answer here.]

Teacher feedback:
**Score: 10/15**
The response attempts to follow the required structure and does answer the question, but not always consistently. The introduction names the character and briefly mentions the situation, yet the thesis statement is vague and does not clearly outline the main points that will be developed. The body paragraphs show some PEEL features, but explanations are often descriptive rather than analytical, and the answer sometimes slips into narration. Language aspects are mentioned but not always accurately analysed. With a clearer thesis and deeper explanation, this could move to a higher band.
"""
    },
    {
        "label": "low_band_example",
        "text": """EXAMPLE
Question: How is the setting important in this extract?
Student answer:
[Put a weak student answer here.]

Teacher feedback:
**Score: 6/15**
This response shows a basic awareness of the setting but does not fully address how it is important to the extract. The introduction is very brief and does not include the title, author, or a clear thesis statement. The paragraphs are loosely organised and do not consistently follow a PEEL structure. There is little direct evidence from the text, and explanations remain general rather than analytical. Language features are not clearly identified or discussed. To improve, the student needs to include a proper introduction, use quotations to support points, and focus on explaining how the setting affects mood, character, or theme.
"""
    },
    # ➜ Add 12–17 more examples in the same structure
]

# -----------------------
#  PROMPT TEMPLATE
# -----------------------
peel_prompt_string = """
You are an experienced IGCSE examiner. Your task is to evaluate a middle school student’s written response using the PEEL structure: Point, Evidence, Explanation, Link.

You will be given several EXAMPLE EVALUATIONS. Each example contains:
- The question
- The student’s answer
- The teacher’s feedback
- The score out of 15

Study these examples carefully and imitate their style, tone, level of strictness, and scoring when evaluating the new answer.

EXAMPLE_EVALUATIONS:
{examples}

Evaluate the new answer against these criteria:

1. POINT — Is the argument clearly stated and directly answering the question?
2. EVIDENCE — Is there relevant, accurate, and specific supporting evidence?
3. EXPLANATION — Does the student explain how the evidence supports the point, showing understanding and analysis?
4. LINK — Does the student connect back to the question or provide a clear transition?

Expectation from the Student's Answer:
There should be one introductory paragraph followed by three body paragraphs and one conclusion paragraph.
The introductory paragraph should first begin with the title of the story,
then the name of the author,
then a short one or two lines about the content of the extract or the content of the scene,
followed by a thesis statement.
The thesis statement should provide a clear answer to the question and outline all the points which will be highlighted in the following essay.
The body paragraphs should follow the PEEL format, but language aspects also require to be mentioned.
In language aspects, the student should identify a literary device or other forms of language used by the author to bring out what they're trying to say.
They can either use the PETAL (Point, Evidence, Technique, Analysis, Link) format for this or provide a separate body paragraph for the same.
If they are providing a separate body paragraph, they will have two paragraphs which explain points and one paragraph which highlights the literary devices.
Then in the conclusion, they must summarize all the points and restate the thesis statement.
Throughout the essay, they should not narrate the story; they need to be specific to the question.
However, evidence must be explained and some content and background may be provided while doing the same.

Evaluation Criteria:
- 10 marks for the content of the answer
- 5 marks for quality of writing (grammar, vocabulary, clarity)
- Total: 15 marks

Now evaluate the NEW answer.

Provide your feedback in paragraph format using the structure below:
- Start with giving a score out of 15 based on overall effectiveness. Output this in bold (Markdown) and add a newline after this line.
- Then add a separate paragraph for each of the following, separated by a blank line:
  - A judgment of how well the PEEL structure is followed.
  - Comments on the strengths in Point, Evidence, Explanation, and Link.
  - 2–3 clear suggestions for improvement (EBI: Even Better If…).
  - A brief summary sentence encouraging improvement.

Tone: Constructive, supportive, and academically appropriate for IGCSE level.
Do not use bullet lists in your feedback paragraphs. Do not provide separate numeric scores for each criterion; only provide one overall score out of 15 at the start.

QUESTION:
{question}

STUDENT_ANSWER:
{student_answer}
"""

prompt = PromptTemplate(
    template=peel_prompt_string,
    input_variables=["examples", "question", "student_answer"],
)

# -----------------------
#  LLM & VECTORSTORE INIT
# -----------------------
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4.1-mini",   # or "gpt-4.1-mini" / "gpt-4o" etc.
        temperature=0
    )

@st.cache_resource
def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    texts = [ex["text"] for ex in EXAMPLE_EVALUATIONS]
    metadatas = [{"label": ex["label"]} for ex in EXAMPLE_EVALUATIONS]

    # In-memory Chroma (no persist_directory) – Streamlit Cloud friendly
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name="peel_examples"
    )
    return vectorstore

def select_examples(vectorstore, student_answer, k=3) -> str:
    """
    Use Chroma to retrieve k similar example evaluations
    based on the student's answer.
    """
    docs = vectorstore.similarity_search(student_answer, k=k)
    examples_text = "\n\n---\n\n".join(doc.page_content for doc in docs)
    return examples_text

def evaluate_answer(question: str, student_answer: str) -> str:
    llm = get_llm()
    vectorstore = get_vectorstore()

    examples_text = select_examples(vectorstore, student_answer, k=3)

    final_prompt = prompt.format(
        examples=examples_text,
        question=question,
        student_answer=student_answer,
    )

    response = llm.invoke(final_prompt)
    # For ChatOpenAI, response is a ChatMessage
    return response.content

# -----------------------
#  STREAMLIT UI
# -----------------------
st.set_page_config(page_title="PEEL Essay Evaluator", layout="wide")
st.title("📝 PEEL Essay Evaluator (IGCSE-style)")
st.write(
    "Paste a literature essay-style answer below and get structured, PEEL-based feedback "
    "with an overall score out of 15."
)

with st.sidebar:
    st.header("About")
    st.markdown(
        "- Uses your marked examples as guidance\n"
        "- Evaluates PEEL structure + language\n"
        "- Designed for IGCSE / middle school essays"
    )
    st.markdown("---")
    st.caption("Make sure your OpenAI API key is set in `st.secrets`.")    

question = st.text_area(
    "Question (prompt given to the student)",
    placeholder="e.g. How does the writer create sympathy for the main character in this extract?",
    height=80,
)

student_answer = st.text_area(
    "Student's answer",
    placeholder="Paste the student's full essay response here...",
    height=260,
)

if st.button("Evaluate Answer", type="primary"):
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Please add it in Streamlit secrets.")
    elif not question.strip() or not student_answer.strip():
        st.warning("Please enter both a question and a student answer.")
    else:
        with st.spinner("Evaluating..."):
            feedback = evaluate_answer(question, student_answer)
        st.subheader("Feedback")
        st.markdown(feedback)
