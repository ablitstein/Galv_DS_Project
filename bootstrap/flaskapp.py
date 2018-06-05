from flask import Flask, render_template
from yelp_scrub import YelpScrub
app = Flask(__name__)

ys = YelpScrub(1)


# home page
#@app.route('/web/')
#def index():
#    return render_template('index.html', title='Hello!')

@app.route('/')
def index():
    return render_template('tables.html')

@app.route('/hello')
def api_articles():
    return 'hello world!'

@app.route('/yelp/<yid>')
def api_yelpest(yid):
    if (id):
        ys.run(yid)
    return 'hello world YELP! '+str(yid)
#"http://0.0.0.0:8000/yelp/1dMU2kz5AhTC6N1W9xwuQ"


#@app.route('/more/')
#def more():
#    return render_template('starter_template.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
