from sklearn.feature_extraction.text import TfidfVectorizer
from odoo import models, api, fields
from sklearn.compose import ColumnTransformer

class TfidfVectorizerEstimator(models.Model):
    _inherit = 'supervised.learning.estimator'

    type = fields.Selection(selection_add=[('tfidf_vectorizer', 'TF-IDF Vectorizer')])
    lowercase = fields.Boolean(string="Lowercase", default=True)
    analyzer = fields.Selection([
        ('word', 'Word'),
        ('char', 'Character'),
        ('char_wb', 'Character n-grams'),
    ], string="Analyzer", default='word')

    def _get_estimator(self):
        self.ensure_one()
        if not self.type == 'tfidf_vectorizer':
            return super()._get_estimator()
        else:
            return ColumnTransformer(
                # tf-idf vectorizer seems to expect a 1d array, so we have to pass the column name as a string, not a list
                # if multiple columns are selected, the transformer will be applied to each column separately
                transformers=[
                    ('tfidf_vectorizer', TfidfVectorizer(lowercase=self.lowercase, analyzer=self.analyzer, max_df=0.5, min_df=5, stop_words='english'), column.name)
                for column in self.column_ids], remainder='passthrough'
            )
        