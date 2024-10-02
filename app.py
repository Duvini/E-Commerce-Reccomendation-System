from flask import Flask, request, render_template
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# Load files
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# Database configuration
app.secret_key = "your_secret_key"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:@localhost/ecom"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define model classes for 'signup' and 'signin' tables
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Recommendations functions# Recommendations functions
def truncate(text, length):
    return text[:length] + "..." if len(text) > length else text


def get_trending_products():
    average_ratings = train_data.groupby(['Name', 'ReviewCount', 'Brand', 'Price', 'ImageURL'], as_index=False).agg({'Rating': 'mean'})
    return average_ratings.sort_values(by='Rating', ascending=False).head(12)

def content_based_recommendations(train_data, item_name, top_n=40):
    search_terms = set(item_name.lower().split())
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2, ngram_range=(1, 2))
    tfidf_matrix_all = tfidf_vectorizer.fit_transform(train_data['Tags'])

    try:
        searched_item_index = train_data[train_data['Name'].str.lower().str.contains('|'.join(search_terms))].index[0]
        searched_item_vector = tfidf_matrix_all[searched_item_index]
    except IndexError:
        print(f"No items found matching any of the terms in '{item_name}'.")
        return pd.DataFrame()

    distances = pairwise_distances(searched_item_vector, tfidf_matrix_all, metric='euclidean').flatten()
    similar_items_df = pd.DataFrame({'Index': train_data.index, 'Distance': distances})
    similar_items_df = similar_items_df[similar_items_df['Index'] != searched_item_index]
    top_similar_items_df = similar_items_df.nsmallest(top_n, 'Distance')

    recommended_item_indices = top_similar_items_df['Index']
    return train_data.loc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'Price', 'ImageURL', 'Rating']]

def collaborative_filtering_recommendations(train_data, target_user_id, top_n):
    # Create a user-item matrix
    user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)

    # Check if the target user ID is in the user-item matrix
    if target_user_id not in user_item_matrix.index:
        return pd.DataFrame()  # Return empty DataFrame if user_id is not found

    # Calculate cosine similarity between users
    user_similarity = cosine_similarity(user_item_matrix)
    
    # Get the index of the target user
    target_user_index = user_item_matrix.index.get_loc(target_user_id)

    # Get similarity scores for the target user
    user_similarities = user_similarity[target_user_index]

    # Get indices of similar users, ignoring the target user
    similar_users_indices = user_similarities.argsort()[::-1][1:]  

    recommended_items = set()
    
    # Loop through similar users
    for user_index in similar_users_indices:
        # Get products rated by the similar user
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        
        # Check which items the target user hasn't rated
        not_rated_by_target_user = (rated_by_similar_user > 0) & (user_item_matrix.iloc[target_user_index] == 0)

        # Update recommended items with new products
        recommended_items.update(user_item_matrix.columns[not_rated_by_target_user][:top_n])

        # Stop if we have enough recommendations
        if len(recommended_items) >= top_n:
            break

    # Filter recommended products from the original train_data
    recommended_products = train_data[train_data['ProdID'].isin(recommended_items)][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Price', 'Rating']]
    
    return recommended_products.head(top_n)  # Ensure you return at most top_n products



# Routes
@app.route("/")
def index():
    trending_products = get_trending_products()
    return render_template('index.html', trending_products=trending_products, truncate=truncate)

@app.route("/signup", methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()

        return render_template('index.html', trending_products=get_trending_products(), truncate=truncate, signup_message='User signed up successfully!')

    return render_template('signup.html')

@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']

        user = Signup.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            return render_template('index.html', trending_products=get_trending_products(), truncate=truncate, signup_message='User signed in successfully!')
        else:
            return render_template('signin.html', error='Invalid credentials')

    return render_template('signin.html')

@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = request.form.get('nbr', default=10, type=int)

        content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

        if not content_based_rec.empty:
            product_image_urls = content_based_rec['ImageURL'].tolist()
            prices = content_based_rec['Price'].tolist()
            # Pass the truncate function to the template
            return render_template('main.html', content_based_rec=content_based_rec, product_image_urls=product_image_urls, prices=prices, truncate=truncate)
        
        return render_template('main.html', message="No recommendations available for this product.")

    return render_template('main.html', truncate=truncate)



@app.route("/recommendations_by_user_id", methods=['POST'])
def recommendations_by_user_id():
    user_id = request.form['user_id']
    try:
        target_user_id = int(user_id)   # Convert user ID to integer
        collaborative_rec = collaborative_filtering_recommendations(train_data, target_user_id, top_n=10)
        
        # Debug: Print the structure of collaborative_rec
        print("Collaborative Recommendations DataFrame:\n", collaborative_rec)
        
        if collaborative_rec.empty:
            return render_template('forYou.html', content_based_rec=pd.DataFrame(), message="No recommendations available for this user.", truncate=truncate)
        
        product_image_urls = collaborative_rec['ImageURL'].tolist()
        prices = collaborative_rec['Price'].tolist()
        return render_template('forYou.html', content_based_rec=collaborative_rec, product_image_urls=product_image_urls, prices=prices, truncate=truncate)
    except ValueError:
        return render_template('forYou.html', content_based_rec=pd.DataFrame(), message="Invalid User ID format.", truncate=truncate)



@app.route("/forYou", methods=['GET'])
def for_you():
    return render_template('forYou.html', content_based_rec=pd.DataFrame())

if __name__ == '__main__':
    app.run(debug=True)
