from odoo import models, api, fields
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

class RandomForestClassifierEstimator(models.Model):
    _inherit = 'supervised.learning.estimator'
    
    type = fields.Selection(selection_add=[('random_forest_classifier', 'Random Forest Classifier')])
    n_estimators = fields.Integer(string="Number of Trees", default=100)
    criterion = fields.Selection([
        ('gini', 'Gini'),
        ('entropy', 'Entropy'),
        ('log_loss', 'Log Loss')
    ], string="Criterion", default='gini')
    max_depth = fields.Integer(string="Max Depth", default=None)
    min_samples_split = fields.Integer(string="Min Samples Split", default=2)
    min_samples_split_pct = fields.Float(string="Min Samples Split (%)", default=0.0)
    min_samples_leaf = fields.Integer(string="Min Samples Leaf", default=1)
    min_samples_leaf_pct = fields.Float(string="Min Samples Leaf (%)", default=0.0)
    min_weight_fraction_leaf = fields.Float(string="Min Weight Fraction Leaf", default=0.0)
    max_features = fields.Selection([
        ('sqrt', 'sqrt'),
        ('log2', 'log2'),
        ('None', 'None'),
    ], string="Max Features", default='sqrt')
    max_leaf_nodes = fields.Integer(string="Max Leaf Nodes", default=None)
    min_impurity_decrease = fields.Float(string="Min Impurity Decrease", default=0.0)
    bootstrap = fields.Boolean(string="Bootstrap", default=True)

    def _get_estimator(self):
        self.ensure_one()
        if self.type != 'random_forest_classifier':
            return super()._get_estimator()
        
        min_samples_split = self.min_samples_split or self.min_samples_split_pct / 100
        min_samples_leaf = self.min_samples_leaf or self.min_samples_leaf_pct / 100

        max_leaf_nodes = self.max_leaf_nodes or None
        max_depth = self.max_depth or None

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap
        )

        if self.pipeline_id.is_multilabel:
            clf = MultiOutputClassifier(clf)

        return clf

    @api.onchange('min_samples_split_pct')
    def _onchange_min_samples_split_pct(self):
        if self.min_samples_split_pct > 0:
            self.min_samples_split = 0

    @api.onchange('min_samples_split')
    def _onchange_min_samples_split(self):
        if self.min_samples_split > 0:
            self.min_samples_split_pct = 0

    @api.onchange('min_samples_leaf_pct')
    def _onchange_min_samples_leaf_pct(self):
        if self.min_samples_leaf_pct > 0:
            self.min_samples_leaf = 0

    @api.onchange('min_samples_leaf')
    def _onchange_min_samples_leaf(self):
        if self.min_samples_leaf > 0:
            self.min_samples_leaf_pct = 0