from flask import Flask
import Citprov
app = Flask(__name__)

@app.route("/")
def hello():
  return 'Hello World'

@app.route("/sayHello")
def sayHello():
  test = Citprov.citprov()
  return test.sayHello()

if __name__ == '__main__':
  app.run()
