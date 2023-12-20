import pandas as pd
import nltk
from nltk.chat.util import Chat, reflections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Baixe os dados necessários para o NLTK
nltk.download('punkt')

# Carregue o arquivo CSV
df = pd.read_csv('database.csv', sep='|')

# Pares de padrão e respostas para o chatbot
pares = []

# Adicione as perguntas e respostas ao chatbot
for _, row in df.iterrows():
    perguntas = [pergunta.lower() for pergunta in row['questions'].split(';')]
    respostas = row['responses']
    intencao = row['intents']
    for pergunta in perguntas:
        pares.append((pergunta, f"{respostas}__{intencao}"))  # Combinamos resposta e intenção em uma string

# Crie um objeto Chat com os pares de padrão e respostas
chatbot = Chat(pares, reflections)


# Função para calcular a similaridade entre duas strings usando TF-IDF
def calcular_similaridade(usuario, perguntas):
    vectorizer = TfidfVectorizer(stop_words=[], lowercase=False)

    # Transforme as mensagens em vetores TF-IDF
    tfidf_matrix = vectorizer.fit_transform([usuario] + perguntas)

    # Calcule a similaridade de cosseno entre a pergunta do usuário e todas as perguntas no conjunto
    similaridades = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    # Obtenha o índice da pergunta mais semelhante
    indice_pergunta_similar = similaridades.argmax()

    # Retorna a resposta e intenção correspondentes à pergunta mais semelhante
    resposta, intencao = pares[indice_pergunta_similar][1].split('__')
    return resposta, intencao


# Função para iniciar o chatbot
def iniciar_chat():
    print("Bem-vindo ao ChatBot. Digite 'adeus' para sair.")
    while True:
        mensagem = input("Você: ")
        if mensagem.lower() == 'adeus':
            print("ChatBot: Até mais!")
            break
        # Calcular a similaridade com todas as perguntas
        resposta, intencao = calcular_similaridade(mensagem.lower(), df['questions'].str.lower().tolist())
        print(f"ChatBot (Intenção: {intencao}): {resposta}")


# Inicie o chatbot
iniciar_chat()
