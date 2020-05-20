from flask import Flask, request, render_template
from model import preprocess
from inference import get_best_match, detect_face, detect_dog

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def run_app():
    if request.method == "GET":
        return render_template("index.html", value="hello")

    if request.method == "POST":
        print(request.files)
        if "file" not in request.files:
            print("File not uploaded")
            return
        file = request.files["file"]
        image = file.read()
        if detect_dog(image_bytes=image)[0]:
            _, breed_name = get_best_match(image_bytes=image)
            index = "Hi dog! You look like a..."

        elif detect_face(image_bytes=image)[0]:
            _, breed_name = get_best_match(image_bytes=image)
            index = "Hello human! You look like a... "

        else:
            message1 = "Check your image!"
            message2 = "It's neither a dog nor a human"
            index, breed_name = message1, message2
        print(f"Class index: {index}")
        print(f"Breed: {breed_name}")
        return render_template(
            "result.html", dog_breed=breed_name, category=index
        )


if __name__ == "__main__":
    app.run(debug=True)
