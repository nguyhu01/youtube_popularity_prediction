import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

class Preprocessor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.tfidf, self.encoder = self._fit_preprocessors()

    def _fit_preprocessors(self):
        # Fit TF-IDF Vectorizer
        tfidf = TfidfVectorizer(max_features=100)
        tfidf.fit(self.data['Title'])

        # Fit One-Hot Encoder
        encoder = OneHotEncoder()
        encoder.fit(self.data[['Channel']])

        return tfidf, encoder

    def transform_input(self, title, channel, published_date):
        # Transform the title using TF-IDF
        title_features = self.tfidf.transform([title]).toarray()

        # Transform the channel using One-Hot Encoder
        channel_encoded = self.encoder.transform([[channel]]).toarray()

        # Process the date
        published_date = pd.to_datetime(published_date)
        day_of_week = published_date.dayofweek
        day_of_year = published_date.dayofyear

        # Combine all features
        combined_features = [title_features[0], channel_encoded[0], [day_of_week, day_of_year]]
        
        # Flatten the combined_features to create a single feature array
        combined_features_flat = [item for sublist in combined_features for item in sublist]

        # Create a DataFrame with proper feature names
        feature_names = self.tfidf.get_feature_names_out().tolist() + \
                    self.encoder.get_feature_names_out().tolist() + \
                    ['day_of_week', 'day_of_year']
        combined_features_df = pd.DataFrame([combined_features_flat], columns=feature_names)
        return combined_features_df


# Example usage:
# preprocessor = Preprocessor('/path/to/your/dataset.csv')
# processed_input = preprocessor.transform_input('Example Title', 'Example Channel', '2023-01-01')
