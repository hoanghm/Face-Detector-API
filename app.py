import os
from flask import Flask, flash, request, redirect, url_for, jsonify, render_template
from werkzeug.utils import secure_filename
from detector import MasterDetector
from time import perf_counter
import matplotlib.pyplot as plt

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
DEBUG = True

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            t0 = perf_counter()
            detector = MasterDetector()
            print("Initialization completed. Elapsed time: {:.2f} seconds".format(perf_counter() - t0))
            filename = secure_filename(file.filename)
            img_path = os.path.dirname(os.path.realpath(__file__)) + "/static/uploads/" + filename
            file.save(img_path)
            print("Image received. Processing image...")
            t1 = perf_counter()
            img, predictions = detector.detect_image(img_path)
            t2 = perf_counter()
            plt.imsave(img_path, img)
            img_link = url_for('static', filename='uploads/' + filename)
            if predictions is not None:
                return render_template('index.html', faces=predictions, img_link=img_link,
                                       time="{:.2f}".format(t2 - t1))
    return render_template('index.html', faces=None)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
