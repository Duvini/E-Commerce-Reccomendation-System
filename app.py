from flask import Flask, request, render_template
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from sklearn.metrics import pairwise_distances


app = Flask(__name__)

# load files===========================================================================================================
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# database configuration---------------------------------------
app.secret_key = "alskdjfwoeieiurlskdjfslkdjf"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:@localhost/ecom"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Define your model class for the 'signup' table
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Define your model class for the 'signup' table
class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)


# Recommendations functions============================================================================================
# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

def get_trending_products():
    # Calculate average ratings and get top rated items
    average_ratings = train_data.groupby(['Name', 'ReviewCount', 'Brand','Price', 'ImageURL'], as_index=False).agg({'Rating': 'mean'})
    top_rated_items = average_ratings.sort_values(by='Rating', ascending=False).head(12)
    return top_rated_items



def content_based_recommendations(train_data, item_name, top_n=40):
    # Split the item_name into words
    search_terms = set(item_name.lower().split())

    # Create a TF-IDF vectorizer for the 'Tags' column
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2, ngram_range=(1, 2))
    
    # Fit the vectorizer to the entire dataset to use for all items
    tfidf_matrix_all = tfidf_vectorizer.fit_transform(train_data['Tags'])

    # Get the TF-IDF vector for the searched item (if found)
    try:
        searched_item_index = train_data[train_data['Name'].str.lower().str.contains('|'.join(search_terms))].index[0]
        searched_item_vector = tfidf_matrix_all[searched_item_index]
    except IndexError:
        print(f"No items found matching any of the terms in '{item_name}'.")
        return pd.DataFrame()

    # Calculate Euclidean distance for all items against the searched item
    distances = pairwise_distances(searched_item_vector, tfidf_matrix_all, metric='euclidean').flatten()

    # Create a DataFrame of similar items
    similar_items_df = pd.DataFrame({
        'Index': train_data.index,
        'Distance': distances
    })

    # Remove the searched item from recommendations and get top_n items with the smallest distances
    similar_items_df = similar_items_df[similar_items_df['Index'] != searched_item_index]
    top_similar_items_df = similar_items_df.nsmallest(top_n, 'Distance')

    # Get the recommended item details
    recommended_item_indices = top_similar_items_df['Index']
    recommended_items_details = train_data.loc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'Price', 'ImageURL', 'Rating']]

    return recommended_items_details





# routes===============================================================================
# List of predefined image URLs
random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
]


@app.route("/")
def index():
    trending_products = get_trending_products()
    custom_prices = {5.79, 42.44, 58.91, 2.3, 26.85, 13.58, 10, 20.99, 15.55, 29.25,20,45}

# If you want to conv
    # Get actual images from the trending_products DataFrame
    product_image_urls = trending_products['ImageURL'].tolist()  # Ensure your DataFrame has this column
    prices = trending_products['Price'].tolist()  # Ensure your DataFrame has this column

    return render_template('index.html', 
                           trending_products=trending_products, 
                           truncate=truncate,
                           product_image_urls=product_image_urls,
                           prices=prices)

@app.route("/main")
def main():
    return render_template('main.html')

# routes
@app.route("/index")
def indexredirect():

    # Get actual images and prices from the trending_products DataFrame
    product_image_urls = trending_products['ImageURL'].tolist()  # Ensure your DataFrame has this column
    prices = trending_products['Price'].tolist() # Ensure your DataFrame has this column

    return render_template('index.html', trending_products=trending_products, 
                           truncate=truncate,
                           product_image_urls=product_image_urls,
                           prices=prices)

@app.route("/signup", methods=['POST','GET'])
def signup():

    
    # Get actual images and prices from the trending_products DataFrame
    product_image_urls = trending_products['ImageURL'].tolist()  # Ensure your DataFrame has this column
    prices = trending_products['Price'].tolist() # Ensure your DataFrame has this column

    if request.method=='POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()

        return render_template('index.html', trending_products=trending_products, 
                           truncate=truncate,
                           product_image_urls=product_image_urls,
                           prices=prices,
                           signup_message='User signed up successfully!'
                               )

# Route for signup page
@app.route('/signin', methods=['POST', 'GET'])
def signin():

    # Get actual images and prices from the trending_products DataFrame
    product_image_urls = trending_products['ImageURL'].tolist()  # Ensure your DataFrame has this column
    prices = trending_products['Price'].tolist() # Ensure your DataFrame has this column

    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']
        new_signup = Signin(username=username,password=password)
        db.session.add(new_signup)
        db.session.commit()

        return render_template('index.html', trending_products=trending_products, 
                           truncate=truncate,
                           product_image_urls=product_image_urls,
                           prices=prices,
                               signup_message='User signed in successfully!'
                               )
@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    content_based_rec = pd.DataFrame()  # Default to an empty DataFrame
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = request.form.get('nbr')

        # If nbr is not provided or is an empty string, return all possible recommendations
        if not nbr or nbr.strip() == '':
            nbr = len(train_data)  # Default to all available products
        else:
            try:
                nbr = int(nbr)
            except ValueError:
                nbr = 10  # Default to 10 if the conversion fails

        content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)


    if not content_based_rec.empty:
        product_image_urls = content_based_rec['ImageURL'].tolist()
        prices = trending_products['Price'].tolist() 

        return render_template('main.html', content_based_rec=content_based_rec, truncate=truncate,
                               product_image_urls=product_image_urls,
                               prices=prices)

    message = "No recommendations available for this product."
    return render_template('main.html', content_based_rec=content_based_rec, message=message, truncate=truncate)


if __name__=='__main__':
    app.run(debug=True)