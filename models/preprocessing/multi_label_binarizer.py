from odoo import models, api, fields
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.compose import ColumnTransformer

class MultiLabelBinarizerEstimator(models.Model):
    _inherit = 'supervised.learning.estimator'

    type = fields.Selection(selection_add=[('multi_label_binarizer', 'Multi Label Binarizer')])

    def _get_estimator(self):
        self.ensure_one()
        if self.type != 'multi_label_binarizer':
            return super()._get_estimator()
        
        return ColumnTransformer(
            transformers=[
                (self.name, MultiLabelBinarizer(), self.column_ids.mapped('name')),
            ], remainder='passthrough'
        )