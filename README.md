# 📧 Classificador e Respondedor de Emails com IA

Este projeto utiliza **Machine Learning** e a API da OpenAI para:  
1. Classificar emails em **Produtivos** ou **Improdutivos**  
2. Sugerir respostas automáticas profissionais com base na classificação  

---

## 🔗 Fine-Tuning
O modelo treinado está disponível no link abaixo:  
👉 [Download do Fine-Tuning](https://drive.google.com/file/d/1RHdZV7XFlJ099vif1Wu0-ucRQqRakmmn/view?usp=sharing)

---

## 🧪 Testes no GPT Playground
Abaixo, um exemplo de como esta configurado o **Prompt no GPT Playground**:  

<p align="center">
  <img width="500" alt="Exemplo GPT Playground" src="https://github.com/user-attachments/assets/1974a860-a0d6-40ae-bbd0-c28a8f679981" />
</p>

---

## 🚀 Tecnologias Utilizadas
- **Python (FastAPI)** → criação da API  
- **HTML, CSS e JavaScript** → interface simples para upload e visualização dos resultados  
- **Machine Learning (Fine-tuning do modelo `neuralmind/bert-base-portuguese-cased`)** → classificação dos emails em produtivos ou improdutivos  
- **OpenAI GPT (API + Playground)** → criação de um prompt no GPT Playground com instruções específicas para geração de respostas.
- **Dotenv** → gerenciamento seguro das variáveis de ambiente  
- **Google Colab / Jupyter Notebook** → treinamento e testes iniciais do modelo de classificação  
- **Ngrok** → deploy temporário e acessível pela web



## 📂 Estrutura do Projeto

```text
projeto-email/
├── 📄 main.py                       # API principal (FastAPI)
├── 📄 classificador.py              # Funções de Machine Learning (fine-tuning BERT)
├── 📄 chamadaGpt.py                 # Integração com a API do GPT Playground
├── 📄 requirements.txt              # Dependências do projeto
├── 📄 README.md                     # Documentação
├── 📁 static/                       # Arquivos estáticos (frontend)
│   └── 📄 index.html                # Interface principal
└── 📁 setfit-email-classificador-v3/  # Modelo treinado (fine-tuning BERT)

