import os

from groq import Groq
from dotenv import load_dotenv, dotenv_values
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY3"),
)
def groq_infr(sys_prompt, message):
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "user",
                "content": message
            }
        ],
        temperature=1,
        max_tokens=200,
        top_p=1,
        stream=False,
        stop=None,
    )

    return completion.choices[0].message.content



