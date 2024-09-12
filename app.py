from flask import Flask, request, render_template
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


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








def content_based_recommendations(train_data, item_name, top_n=10):
    # Split the item_name into words
    search_terms = set(item_name.lower().split())

    # Create a function to check for whole word matches using word boundaries
    def word_match(name, terms):
        name_tokens = set(re.findall(r'\b\w+\b', name.lower()))
        return bool(name_tokens.intersection(terms))
    
    # Find items where the whole word matches any of the search terms in the 'Name' column
    matching_indices = train_data.index[train_data['Name'].apply(lambda name: word_match(name, search_terms))]

    if len(matching_indices) == 0:
        print(f"No items found matching any of the terms in '{item_name}'.")
        return pd.DataFrame()

    # Filter the dataframe to include only matching items
    filtered_data = train_data.loc[matching_indices]

    if filtered_data.empty:
        return pd.DataFrame()

    # Create a TF-IDF vectorizer for the 'Tags' column with adjustments to extract meaningful words
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2, ngram_range=(1, 2))

    # Apply TF-IDF vectorization to the 'Tags' column
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(filtered_data['Tags'])

    # Calculate cosine similarity between items based on their 'Tags'
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    # Dictionary to store results
    all_similar_items = []

    # Map the original indices to their positions in the filtered data
    index_map = {original_index: pos for pos, original_index in enumerate(filtered_data.index)}

    for idx, item_index in enumerate(filtered_data.index):
        # Use the mapped position in the cosine similarity matrix
        matrix_pos = index_map[item_index]
        
        # Ensure the matrix_pos is within bounds
        if matrix_pos >= len(cosine_similarities_content):
            print(f"Index {matrix_pos} is out of bounds for cosine similarity matrix.")
            continue

        similar_items = list(enumerate(cosine_similarities_content[matrix_pos]))
        similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
        top_similar_items = similar_items[1:top_n+1]  # Exclude the item itself
        
        for sim_index, score in top_similar_items:
            original_index = filtered_data.index[sim_index]
            all_similar_items.append((original_index, score))

    # Create a DataFrame from the collected similar items
    all_similar_items_df = pd.DataFrame(all_similar_items, columns=['Index', 'Score'])

    # Remove duplicates and get the top_n recommendations
    all_similar_items_df = all_similar_items_df.drop_duplicates(subset='Index')
    top_similar_items_df = all_similar_items_df.nlargest(top_n, 'Score')

    # Get the recommended item details
    recommended_item_indices = top_similar_items_df['Index']
    recommended_items_details = train_data.loc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

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
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html',trending_products=trending_products.head(8),truncate = truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price = random.choice(price))

@app.route("/main")
def main():
    return render_template('main.html')

# routes
@app.route("/index")
def indexredirect():
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price=random.choice(price))

@app.route("/signup", methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()

        # Create a list of random image URLs for each product
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                               signup_message='User signed up successfully!'
                               )

# Route for signup page
@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']
        new_signup = Signin(username=username,password=password)
        db.session.add(new_signup)
        db.session.commit()

        # Create a list of random image URLs for each product
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                               signup_message='User signed in successfully!'
                               )
@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    content_based_rec = pd.DataFrame()  # Default to an empty DataFrame
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = int(request.form.get('nbr'))
        content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

    # Create a list of random image URLs for each recommended product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

    if content_based_rec.empty:
        message = "No recommendations available for this product."
        return render_template('main.html', content_based_rec=content_based_rec, message=message, truncate=truncate,
                               random_product_image_urls=random_product_image_urls,
                               random_price=random.choice(price))
    else:
        return render_template('main.html', content_based_rec=content_based_rec, truncate=truncate,
                               random_product_image_urls=random_product_image_urls,
                               random_price=random.choice(price))


if __name__=='__main__':
    app.run(debug=True)