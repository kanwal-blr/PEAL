import os
import random
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate


# =========================
#  CONFIG: OPENAI KEY
# =========================

import openai
#os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

api_key = os.getenv("OPENAI_API_KEY")  # locally: export OPENAI_API_KEY="sk-..."
if not api_key:
    # On Streamlit Cloud, secrets are also exposed as env vars, but in case:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = api_key
    except Exception:
        pass

# =========================
#  EXAMPLE EVALUATIONS
# =========================
# TODO: Replace placeholder contents with your real examples (15–20 entries ideally).
EXAMPLE_EVALUATIONS = [
    {
        "label": "high_band_example",
        "band": "13–15",
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
        "band": "9–12",
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
        "band": "0–8",
        "text": """EXAMPLE
Question: How is the setting important in this extract?
Student answer:
[Put a weak student answer here.]

Teacher feedback:
**Score: 6/15**
This response shows a basic awareness of the setting but does not fully address how it is important to the extract. The introduction is very brief and does not include the title, author, or a clear thesis statement. The paragraphs are loosely organised and do not consistently follow a PEEL structure. There is little direct evidence from the text, and explanations remain general rather than analytical. Language features are not clearly identified or discussed. To improve, the student needs to include a proper introduction, use quotations to support points, and focus on explaining how the setting affects mood, character, or theme.
"""
    },
    # ➜ Add more examples here in the same format
]

# =========================
#  PROMPT TEMPLATE
# =========================

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

# =========================
#  LLM & VECTORSTORE
# =========================

@st.cache_resource
def get_llm(model_name: str, temperature: float):
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )

@st.cache_resource
def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    texts = [ex["text"] for ex in EXAMPLE_EVALUATIONS]
    metadatas = [{"label": ex["label"], "band": ex["band"]} for ex in EXAMPLE_EVALUATIONS]

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name="peel_examples",
    )
    return vectorstore

def select_examples(vectorstore, student_answer: str, k: int = 3) -> tuple[str, list]:
    docs = vectorstore.similarity_search(student_answer, k=k)
    examples_text = "\n\n---\n\n".join(d.page_content for d in docs)
    return examples_text, docs

def evaluate_answer(question: str, student_answer: str, model_name: str, temperature: float, k_examples: int):
    llm = get_llm(model_name, temperature)
    vectorstore = get_vectorstore()

    examples_text, docs_used = select_examples(vectorstore, student_answer, k=k_examples)

    final_prompt = prompt.format(
        examples=examples_text,
        question=question,
        student_answer=student_answer,
    )

    response = llm.invoke(final_prompt)
    return response.content, docs_used


# =========================
#  STREAMLIT UI
# =========================

st.set_page_config(
    page_title="PEEL Essay Evaluator",
    layout="wide",
    page_icon="📝",
)

# ---- TOP BAR ----
st.markdown(
    """
    <style>
    .big-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #666666;
        margin-top: 0.25rem;
        margin-bottom: 1.5rem;
    }
    .result-box {
        border-radius: 0.75rem;
        padding: 1.2rem 1.4rem;
        border: 1px solid #e0e0e0;
        background-color: #fafafa;
    }
    .pill {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        background-color: #eef2ff;
        color: #3730a3;
        font-size: 0.7rem;
        margin-right: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">📝 PEEL Essay Evaluator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">IGCSE-style feedback with PEEL and language focus. Paste a student essay, choose settings, and generate detailed, rubric-based comments.</div>',
    unsafe_allow_html=True,
)

# Warn if API key missing
if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ OpenAI API key not found. Set `OPENAI_API_KEY` as an environment variable or Streamlit secret.")

# ---- LAYOUT: two main columns ----
left_col, right_col = st.columns([2.5, 1.5])

with left_col:
    st.subheader("✏️ Student Response")

    question = st.text_area(
        "Question / Prompt",
        placeholder="e.g. How does the writer create sympathy for the main character in this extract?",
        height=150,
    )

    student_answer = st.text_area(
        "Student's Answer",
        placeholder="Paste the student's full essay response here...",
        height=250,
    )

    run_button = st.button("🔍 Evaluate Answer", type="primary", use_container_width=True)

with right_col:
    st.subheader("⚙️ Settings & Model")

    model_name = st.selectbox(
        "Model",
        options=[
            "gpt-5",
            "gpt-4.1-mini",
            "gpt-4o-mini",
        ],
        index=0,
        help="Use a stronger model for more nuanced feedback; smaller models are cheaper/faster.",
    )

    k_examples = st.slider(
        "Number of examples to guide evaluation",
        min_value=1,
        max_value=min(5, len(EXAMPLE_EVALUATIONS)),
        value=min(3, len(EXAMPLE_EVALUATIONS)),
        step=1,
        help="How many marked examples should the model see before evaluating the new answer?",
    )

    temperature = st.slider(
        "Creativity (temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Keep this low for consistent, rubric-like marking.",
    )

    st.markdown("### ℹ️ Guidance")
    st.markdown(
        "- Expects intro, 3 body paragraphs, and a conclusion.\n"
        "- Body paragraphs should follow PEEL / PETAL.\n"
        "- Focus on analysis, not narration."
    )

    with st.expander("View raw example bands"):
        for ex in EXAMPLE_EVALUATIONS:
            st.markdown(f"**{ex['label']}** (Band {ex['band']})")
            st.code(ex["text"][:500] + ("...\n[truncated]" if len(ex["text"]) > 500 else ""), language="markdown")

# ---- EVALUATION + RESULTS ----
def copy_to_clipboard(text: str):
    # Escape backticks so Markdown doesn’t break JS
    safe_text = text.replace("`", "\\`")
    copy_js = f"""
    <script>
    navigator.clipboard.writeText(`{safe_text}`);
    </script>
    """
    st.markdown(copy_js, unsafe_allow_html=True)
    st.toast("Copied to clipboard!")


    # The button users see
    if st.button(label, use_container_width=True):
        clipboard_js = f"""
        navigator.clipboard.writeText(`{text}`);
        """
        st.experimental_js(clipboard_js)
        st.toast("Copied to clipboard!")  # Nice visual confirmation

if run_button:
    if not question.strip() or not student_answer.strip():
        st.error("Please enter both a **Question** and a **Student's Answer** before evaluating.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key missing. Set `OPENAI_API_KEY` in your environment or Streamlit secrets.")
    else:
        with st.spinner("Evaluating answer using PEEL criteria..."):
            feedback, docs_used = evaluate_answer(
                question=question,
                student_answer=student_answer,
                model_name=model_name,
                temperature=temperature,
                k_examples=k_examples,
            )

        # st.markdown("### ✅ Evaluation Result")
        # st.markdown('<div class="result-box">', unsafe_allow_html=True)
        # st.markdown(feedback)
        # st.markdown("</div>", unsafe_allow_html=True)

        # with st.expander("🔎 Examples used for this evaluation"):
        #     st.caption("The model was guided by these marked examples (retrieved via similarity search):")
        #     for i, d in enumerate(docs_used, start=1):
        #         label = d.metadata.get("label", f"example_{i}")
        #         band = d.metadata.get("band", "?")
        #         st.markdown(f'<span class="pill">Band {band}</span> **{label}**', unsafe_allow_html=True)
        #         st.code(d.page_content[:800] + ("...\n[truncated]" if len(d.page_content) > 800 else ""), language="markdown")

        st.markdown("### ✅ Evaluation Result")
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(feedback)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- ACTION BUTTONS: Copy + Download ---
        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            st.download_button(
                "⬇️ Download feedback as .txt",
                data=feedback,
                file_name="peel_feedback.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with btn_col2:
            # Simple JS-based copy-to-clipboard button
            # safe_feedback = feedback.replace("`", "\\`")
            # copy_button_html = f"""
            # <button onclick="navigator.clipboard.writeText(`{safe_feedback}`)">
            #     📋 Copy feedback
            # </button>
            # """
           
            if st.button("📋 Copy feedback", use_container_width=True):
                copy_to_clipboard(feedback)

            # st.markdown(copy_button_html, unsafe_allow_html=True)

        # --- SHOW WHICH EXAMPLES WERE USED ---
        with st.expander("🔎 Examples used for this evaluation"):
            st.caption("The model was guided by these marked examples (retrieved via similarity search):")
            for i, d in enumerate(docs_used, start=1):
                label = d.metadata.get("label", f"example_{i}")
                band = d.metadata.get("band", "?")
                st.markdown(
                    f'<span class="pill">Band {band}</span> **{label}**',
                    unsafe_allow_html=True,
                )
                st.code(
                    d.page_content[:800] + ("...\n[truncated]" if len(d.page_content) > 800 else ""),
                    language="markdown",
                )
# Footer
st.markdown("---")
st.caption("PEEL & PETAL essay evaluator • Designed for IGCSE-style literature responses.")



