import os

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
import PyPDF2
from sympy import false

from classificador import classificar_email
from setfit import SetFitModel
from chamadaGpt import gerarResposta
app = FastAPI()
modelo = SetFitModel.from_pretrained("./setfit-email-classificador-v3")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve o arquivo HTML principal da sua aplicação."""
    return FileResponse('static/index.html')


@app.post("/upload")
async def endpoint(file: UploadFile | None = None, email_text: str = Form("")):
    text = ""
    valid = False

    if file:
        valid = True
        if file.filename.endswith(".pdf") or file.filename.endswith(".txt"):
            #pdf
            if file.filename.endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(file.file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""

            #txt
            elif file.filename.endswith(".txt"):
                content = await file.read()
                text = content.decode("utf-8")



    elif email_text:
        text = email_text
        #print(text)
        valid = True

    if valid:
        result = classificar_email(text, modelo)
        # print(result)
        print("chamando api do gpt para gerar resposta")
        text = f"Classificacao [{result['label']}] {gerarResposta(text)}"

    else:
        text = "Nenhum dado enviado"

    return text
