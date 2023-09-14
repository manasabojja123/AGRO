from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.image import load_img
from PIL import Image
from keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
model = load_model('model3.h5')

classes = ['Fresh Fruit', 'Fresh Fruit', 'Fresh Fruit', 'Rotten Fruit', 'Rotten Fruit', 'Rotten Fruit']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
    
        image = Image.open(file)
        image = image.resize((64, 64))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
    
        plt.subplot(1, 2, 1)
        plt.imshow(image[0])
        plt.axis("off")
        plt.savefig('static/plot.png')
        plt.close()
        
        result = model.predict(image)
        predicted_class_index = np.argmax(result[0])
        prediction = classes[predicted_class_index]
    
        return render_template('1.html', prediction=prediction)
    
    return render_template('1.html')

if __name__ == '__main__':
    app.run(debug=True)