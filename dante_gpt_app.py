from flask import Flask, request, jsonify, render_template
import torch
from gpt import Model2
from gpt_configurations import GPTConfig2


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

    context = torch.tensor(config.str2int(user_input), device=config.device).view(1, -1)
    response = model.generate(context, device=config.device, max_new_tokens=200,
                              min_new_tokens=15, context_size=config.context_size,
                              idx_to_char=config.int_to_str)
    response = response.replace("\n", "<br>")
    return jsonify({'response': response})


if __name__ == '__main__':
    # Load the configuration
    config = GPTConfig2()
    config.device = 'cpu'
    # Load model
    path = "models_heroku/model2_smaller2.pth"
    model = Model2(config)
    model.load_state_dict(torch.load(path, map_location=config.device))
    model.eval()  # Set the model to evaluation mode, not training
    # Launch the app
    app.run(host='0.0.0.0', port=8080, debug=False)
