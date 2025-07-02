#!/usr/bin/env python3

from flask import Flask, render_template, send_from_directory

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/health')
def health():
    return {'status': 'Frontend server running'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)