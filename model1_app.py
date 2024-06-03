from flask import Flask, request, jsonify, render_template
import torch
from model1_chatbot import Chatbot
from gpt import GPTConfig2Small, Model2  #GPTConfig1, Model1

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    user_input = data['initial_string']

    if len(user_input) > 256:
        return jsonify({'error': 'Input exceeds the maximum limit of 256 characters.'}), 400

    context = torch.tensor(chatbot.str2int(user_input), device=config.device).view(1, -1)
    indices = chatbot.model.generate(context, device=config.device, max_new_tokens=500,
                                     context_size=config.context_size)[0].tolist()
    output_text = chatbot.int2str(indices)
    response = output_text[len(user_input):]  # remove the input text
    # end the output at the right-most full stop
    response = response[:response.rfind(".")+1]
    return jsonify({'response': response})


if __name__ == '__main__':
    # Load the configuration and model
    config = GPTConfig2Small()
    config.device = 'cpu'
    config.n_emb = 100
    chatbot = Chatbot(model=Model2, config=config, path="models_heroku/model2_smaller2.pth")

    app.run(host='0.0.0.0', port=8080)


