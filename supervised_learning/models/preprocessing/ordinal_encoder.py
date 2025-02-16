from odoo import models, api, fields
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

class OrdinalEncoderEstimator(models.Model):
    _inherit = 'supervised.learning.estimator'

    def _get_estimator(self):
        self.ensure_one()
        if self.type != 'ordinal_encoder':
            return super()._get_estimator()
        
        return ColumnTransformer(
            transformers=[
                (self.name, OrdinalEncoder(), self.column_ids.mapped('name')),
        ], remainder='passthrough'
        )
        