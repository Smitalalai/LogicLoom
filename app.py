import pickle
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = Flask(__name__)
train_data = pd.read_csv(r"C:\\Users\\srini\\Downloads\\train_set.csv")
X = train_data['Article_content']
y = train_data['Article_type']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
# Load the model
with open("model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)


    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(X_train)
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        text = request.form['text']
        text_vectorized = vectorizer.transform([text])
        if len(text_vectorized.shape) == 1:
            text_vectorized = text_vectorized.reshape(1, -1)

        prediction = model.predict(text_vectorized)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
