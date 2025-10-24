import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

class ContentRecommender():
    def __init__(self, movies_df, train_data, tfidf_max_features=20000, svd_dim=256, normalize_emb=True):
        self.train_data = train_data
        
        movies_df = movies_df.copy()
        movies_df['titlegenres'] = movies_df['title'] + ' ' + movies_df['genres']
        movies_df['full'] = movies_df['titlegenres'] + ' ' + movies_df['description'].fillna('')
    
        def compute_tfidf(corpus):
            tfidf_vectorizer = TfidfVectorizer(
                tokenizer=lambda s: re.findall(r'\w+|\S', s.lower()),
                lowercase=False,
                max_features=tfidf_max_features,
                ngram_range=(1,2),
                stop_words='english'
            )
            X = tfidf_vectorizer.fit_transform(corpus)
            svd = TruncatedSVD(n_components=svd_dim, random_state=42)
            X_reduced = svd.fit_transform(X)
            if normalize_emb:
                X_reduced = normalize(X_reduced)
            return X_reduced
    
        tfidf_full = compute_tfidf(movies_df['full'].tolist())
    
        self.item_emb_full = create_bert_embeddings(movies_df['full'].tolist(), 'full')
    
        self.emb_full = np.hstack([self.item_emb_full, tfidf_full])
        if normalize_emb:
            self.emb_full = normalize(self.emb_full)
    
        self.item_to_idx = {iid: idx for idx, iid in enumerate(movies_df['item_id'])}

    def score(self, user_id, movie_id):
        if movie_id not in self.item_to_idx:
            return 0.0
    
        user_ratings = self.train_data[self.train_data["user_id"] == user_id]
        if len(user_ratings) == 0:
            return 0.0
    
        item_indices = [
            self.item_to_idx[iid] for iid in user_ratings["item_id"]
            if iid in self.item_to_idx
        ]
        if len(item_indices) == 0:
            return 0.0
    
        weights = user_ratings[user_ratings["item_id"].isin(self.item_to_idx.keys())]["rating"].values
        item_embs = self.emb_full[item_indices]
    
        user_profile = np.average(item_embs, axis=0, weights=weights)
    
        user_profile_norm = user_profile / (np.linalg.norm(user_profile) + 1e-9)
        target_emb = self.emb_full[self.item_to_idx[movie_id]]
        target_emb_norm = target_emb / (np.linalg.norm(target_emb) + 1e-9)
    
        score = np.dot(user_profile_norm, target_emb_norm)
    
        return float(score)


content_recommender = ContentRecommender(movies, train_data)

# Print example score
example_movie_id = movies['item_id'].iloc[0]
example_user_id = train_data['user_id'].iloc[0]

print(content_recommender.score(example_user_id, example_movie_id))