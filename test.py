from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import torch
from model1_chatbot import Chatbot
from gpt import GPTConfig2Small, Model2


app = Flask(__name__)
CORS(app)


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
    response = chatbot.model.generate(context, device=config.device, max_new_tokens=200,
                                      min_new_tokens=20, context_size=config.context_size,
                                      idx_to_char=chatbot.int_to_str)  #[0].tolist()
    # output_text = chatbot.int2str(indices)
    # response = output_text[len(user_input):]  # remove the input text
    # end the output at the right-most full stop
    # response = response[:response.rfind(".")+1]
    response = response.replace("\n", "<br>")
    return jsonify({'response': response})


if __name__ == '__main__':
    # Load the configuration and model
    config = GPTConfig2Small()
    config.device = 'cpu'
    config.n_emb = 100
    chatbot = Chatbot(model=Model2, config=config, path="models/model2_smaller2.pth")

    app.run(host='0.0.0.0', port=8080, debug=False)


