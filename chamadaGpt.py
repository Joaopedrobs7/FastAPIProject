from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def gerarResposta(text):
  email_text = text
  #client = OpenAI(api_key=api_key)
  response = client.responses.create(
      # Aqui você referencia o Prompt salvo no Playground
      prompt={
          "id": "pmpt_68b4dbecfc748193836a0af55df0c3330c3c85b433227088",
          "version": "2"
      },
      # O texto que vai preencher o input do prompt
      input=[
          {
              "role": "user",
              "content": email_text
          }
      ]
  )

  return response.output_text
