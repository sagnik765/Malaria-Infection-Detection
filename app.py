from flask import Flask, render_template, request
from keras.models import model_from_json
from keras.preprocessing import image

app = Flask(__name__)

dic = {0 : 'Uninfected', 1 : 'Parasitized'}

json = 'models/Model1.json'
json_file = open(json, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
weights = 'models/Model1_weights.h5'
model.load_weights(weights)

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(128,128))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 128,128,3)
	p = model.predict(i)
	val = 0
	if p[0][1] > p[0][0] :
		val = 1

	return dic[val]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ == '__main__':
	app.debug = True
	app.run()