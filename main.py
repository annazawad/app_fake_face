from flask import Flask
from app import views
app = Flask(__name__)

#app.add_url_rule(rule='/', endpoint = 'home', view_func=views.index)
app.add_url_rule(rule='/', endpoint = 'home', view_func=views.app)
app.add_url_rule(rule='/fake/',
                 endpoint = 'fake',
                 view_func=views.fake_app,
                 methods=['GET','POST'])


if __name__ == '__main__':
    app.run(debug=True)