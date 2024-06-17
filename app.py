from flask import Flask, render_template
import os
import json
import os
from datetime import timedelta

app = Flask(__name__,static_folder='./static')
app.secret_key = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=100)
PORT_NUM = 7860



@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,port=PORT_NUM)