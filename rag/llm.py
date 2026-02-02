from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import LLM_MODEL, OPENAI_API_KEY, TENANT_ID, VLLM_BASE_URL

SYSTEM_PROMPT = (
    f"A {TENANT_ID} nevű vállalat asszisztense vagy, válaszolj röviden az ügyfél kérdéseire. "
    "Használd a kontextust, ha nem találod benne a megfelelő választ mondd, hogy nincs információd erről.\n\n"
    "FONTOS: A válaszod hangosan lesz felolvasva, ezért a telefonszámokat kétjegyű csoportokban írd ki szöveggel, "
    "vesszőkkel elválasztva a szünetek miatt. "
    "Példa: +36302091987 helyett írd: plusz harminchat, harminc, kettő nulla, kilenc tizenkilenc, nyolcvanhét. "
    "Ne használj számjegyeket a telefonszámokban, mindig írd ki betűkkel vesszőkkel tagolva!"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Kérdés: {question}\n\nKontextus:\n{context}"),
    ]
)

llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=VLLM_BASE_URL,
    api_key=OPENAI_API_KEY,
    temperature=0.5,
    top_p=0.9,
    timeout=60,
    max_retries=2,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

__all__ = ["SYSTEM_PROMPT", "prompt", "llm"]
