from gradio_client import Client


# client = Client("Nymbo/Phi-3-5-Mini-Instruct")

def hf_spaces_infr(sys_prompt, message):

    result = client.predict(
            message=message,
            system_prompt=sys_prompt,
            temperature=0.8,
            max_new_tokens=1024,
            top_p=1,
            top_k=20,
            penalty=1.2,
            api_name="/chat")
    return result