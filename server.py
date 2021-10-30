from flask import Flask, request, jsonify
from predict import predict_original_name, init

app = Flask(__name__)
tokenizer, max_news_len, model_lstm, classes = init()


@app.route("/", methods=['POST'])
def index():
    content = request.json
    try:

        max_value, class_num, predicted_name = predict_original_name(
            content['name'],
            tokenizer,
            max_news_len,
            model_lstm,
            classes
        )

        return jsonify({
            "success": True,
            "max_value": str(max_value),
            "class_num": str(class_num),
            "predicted_name": predicted_name
        })
    except Exception as e:
        print(e)

    return jsonify({"success": False})


if __name__ == "__main__":
    app.run(debug=True)
