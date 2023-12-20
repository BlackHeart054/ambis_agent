import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.chat.util import Chat, reflections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

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


# Endpoint para receber perguntas e retornar respostas com intenções
@app.route('/chat', methods=['POST'])
def responder_perguntas():
    data = request.get_json()
    pergunta = data['input']
    resposta, intencao = calcular_similaridade(pergunta.lower(), df['questions'].str.lower().tolist())
    return jsonify({'response': resposta, 'intent': intencao})


# Endpoint para adicionar novas intenções, perguntas e respostas ao CSV
@app.route('/create', methods=['POST'])
def adicionar_intencao():
    global df  # Torna df global para poder modificá-lo dentro da função
    data = request.get_json()
    nova_intencao = data['Intent']
    nova_pergunta = data['Questions']
    nova_resposta = data['Responses']

    # Adicione a nova intenção, pergunta e resposta ao CSV
    df = pd.concat(
        [df, pd.DataFrame({'intents': [nova_intencao], 'questions': [nova_pergunta], 'responses': [nova_resposta]})],
        ignore_index=True)
    df.to_csv('database.csv', sep='|', index=False)

    return jsonify({'mensagem': 'Intenção, pergunta e resposta adicionadas com sucesso!'})


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
