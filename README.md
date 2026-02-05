# 📘 Roadmap de Projetos em Data Science & AI Engineering

Este repositório reúne **meus projetos presentes e futuros** em Ciência de Dados e Inteligência Artificial, organizados para demonstrar domínio técnico, arquitetura profissional, aplicações reais e deploy em produção.

O objetivo final é conquistar minha primeira vaga atuando entre:
- Cientista de Dados
- Machine Learning Engineer
- AI Engineer (LLMs e aplicações inteligentes)

---

## 🚀 Objetivo Geral

Construir um portfólio sólido que demonstre:

- Resolução de problemas reais com Machine Learning
- Construção de aplicações com IA e LLMs
- Deploy em ambiente real (Streamlit, Docker, AWS)
- Pipelines limpos, reprodutíveis e modulares
- Integração entre Data Science e Engenharia de IA
- Arquitetura de aplicações inteligentes

---

# 🧠 Tecnologias

### **Linguagens**
- Python
- SQL

### **Machine Learning / Data Science**
- Scikit-learn
- Modelos supervisionados e não supervisionados
- Métricas de avaliação
- Hyperparameter tuning
- Feature Engineering
- Redução de dimensionalidade

### **IA & LLMs**
- NLP clássico
- Transformers
- LangChain
- RAG (Retrieval-Augmented Generation)
- LLM APIs
- Prompt Engineering

### **Deep Learning**
- PyTorch

### **Deploy & Engenharia**
- Streamlit
- Docker
- AWS
- Git + GitHub
- APIs REST

---

# 🤖 Skills de AI Engineer (Em Desenvolvimento)

### Arquitetura de Sistemas com IA
- Design de pipelines com LLM
- Sistemas com RAG
- Orquestração com LangChain
- Construção de agentes
- Integração LLM + banco de dados

### Engenharia de Aplicações Inteligentes
- APIs com FastAPI
- Estruturação modular
- Observabilidade de modelos
- Monitoramento de outputs
- Versionamento de modelos

### Performance & Produção
- Quantização de modelos
- Caching de embeddings
- Vetorização e bancos vetoriais
- Otimização de custo em LLM

---

# 📂 Projetos e Melhorias Planejadas

---

## 1️⃣ Customer Churn — Classificação

**Resumo:**  
Modelo para prever clientes com risco de cancelamento em banco.

**Ferramentas:**  
Python, Pandas, Scikit-learn, Matplotlib, Docker, AWS

**Deploy:**
- Docker
- AWS ECS
- Endpoint REST

---

## 2️⃣ Loan Approval — Classificação + Regressão

**Resumo:**  
Previsão de aprovação de empréstimo e taxa de juros estimada.

**Ferramentas:**  
Python, Pandas, Scikit-learn, Streamlit

**Deploy:**
- Streamlit Cloud

---

## 3️⃣ Segmentação de Clientes Ecommerce — Clusterização

**Resumo:**  
Segmentação com KMeans para estratégias de marketing e retenção.

**Ferramentas:**  

- Python, Scikit-learn, Seaborn
---

## 4️⃣ NLP Reviews com LLM (Ollama + NLP)

**Resumo:**  
Sistema de análise de reviews textuais utilizando NLP clássico + LLM local via Ollama.

**Funcionalidades:**
- Pré-processamento NLP
- Word2Vec / embeddings
- Classificação de sentimento
- Geração de insights automáticos
- Análise semântica de reviews

**Ferramentas:**
- Python
- NLP clássico
- Ollama
- LLM local
- Streamlit


**Deploy:**
- Streamlit
---

## 5️⃣ Sistema FAQ Inteligente — RAG + LangChain

**Resumo:**  
Sistema que responde perguntas automaticamente utilizando base de conhecimento com RAG.

**Funcionalidades:**
- Indexação de documentos
- Banco vetorial
- Recuperação contextual
- Geração de respostas com LLM
- Interface interativa

**Ferramentas:**
- Python
- LangChain
- Embeddings
- LLM
- Streamlit ou API

**Deploy planejado:**

---

# NLP Roadmap 

# 🧠 Roadmap de Conceitos — NLP Clássico → NLP Moderno → RAG

Roadmap conceitual focado nas habilidades mais exigidas em vagas de Data Science, Machine Learning e NLP aplicado a negócios.  
Organizado em ordem crescente de complexidade técnica.

---

## 🟢 NÍVEL 1 — Fundamentos de Texto e NLP Clássico

### 📌 Conceitos Básicos de Texto
- Tokenization
- Lowercasing / Normalização
- Stopwords
- Stemming
- Lemmatization
- N-grams
- Bag of Words (BoW)

### 📌 Representação Numérica de Texto
- One-hot encoding para texto
- Term Frequency (TF)
- TF-IDF
- Sparse Matrices
- Feature Engineering em texto

### 📌 Estatística aplicada ao NLP
- Frequência de palavras
- Distribuição Zipf
- Similaridade de cosseno
- Distância euclidiana
- Jaccard similarity

---

## 🟡 NÍVEL 2 — Machine Learning aplicado a NLP

### 📌 Modelos Clássicos para NLP
- Naive Bayes (Multinomial / Bernoulli)
- Logistic Regression
- SVM para texto
- Random Forest em embeddings
- Gradient Boosting em features textuais

### 📌 Problemas comuns em NLP
- Classificação de Sentimento
- Spam Detection
- Topic Classification
- Named Entity Recognition (NER) clássico
- Detecção de churn via texto

### 📌 Avaliação de Modelos
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

## 🟠 NÍVEL 3 — Representações Semânticas (Embeddings)

### 📌 Word Embeddings Clássicos
- Word2Vec
  - CBOW
  - Skip-gram
- GloVe
- FastText

### 📌 Conceitos Importantes
- Espaço vetorial semântico
- Similaridade semântica
- Analogias vetoriais
- Contexto local vs global
- Janela de contexto

### 📌 Sentence Embeddings
- Doc2Vec
- Sentence Transformers
- Embeddings densos vs esparsos

---

## 🔴 NÍVEL 4 — Deep Learning para NLP

### 📌 Modelos Sequenciais
- RNN
- LSTM
- GRU
- Problema de Vanishing Gradient

### 📌 Attention Mechanism
- Self-attention
- Multi-head attention
- Positional Encoding

### 📌 Transformers
- Encoder
- Decoder
- Encoder-Decoder
- Masked Language Modeling
- Next Sentence Prediction

---

## 🟣 NÍVEL 5 — Large Language Models (LLMs)

### 📌 Conceitos Fundamentais
- Token embeddings
- Context window
- Prompt engineering
- Zero-shot / Few-shot
- Fine-tuning
- Instruction tuning
- Alignment

### 📌 Inferência
- Temperature
- Top-k / Top-p sampling
- Hallucinations
- Latência e custo computacional

---

## 🔵 NÍVEL 6 — Busca Semântica e Vector Databases

### 📌 Embeddings para Busca
- Semantic search
- Dense retrieval
- Similaridade vetorial
- Approximate Nearest Neighbors (ANN)

### 📌 Vector Databases
- FAISS
- ChromaDB
- Weaviate
- Pinecone
- Milvus

### 📌 Conceitos Técnicos
- Indexação vetorial
- Chunking de documentos
- Embedding pipelines
- Metadata filtering

---

## ⚫ NÍVEL 7 — RAG (Retrieval-Augmented Generation)

### 📌 Arquitetura RAG
- Pipeline de ingestão
- Chunking estratégico
- Embedding de documentos
- Retriever
- Generator
- Context injection

### 📌 Estratégias Avançadas
- Hybrid search (keyword + vector)
- Reranking
- Multi-query retrieval
- Conversational memory
- Grounding

### 📌 Avaliação de RAG
- Faithfulness
- Relevância
- Context precision
- Retrieval recall
- Hallucination rate

---

## 🧩 Conceitos Extras Muito Pedidos em Vagas

- Data Leakage em NLP
- Pipeline de ML para texto
- Deploy de modelos NLP
- APIs de inferência
- Monitoramento de modelos
- Versionamento de dados e modelos
- Experiment tracking
- MLOps para NLP
- Prompt versioning
- Observabilidade de LLM

---

## 🎯 Resultado Esperado

Ao dominar estes conceitos, você estará preparado para:

- Projetos clássicos de NLP
- NLP moderno com Transformers
- Sistemas RAG corporativos
- Aplicações com LLMs
- Projetos reais de Data Science com texto

---




# 📄 Status

Este repositório será atualizado constantemente com novos projetos, melhorias, deploys e evoluções na área de Data Science e Engenharia de IA.
