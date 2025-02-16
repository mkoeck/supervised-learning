from odoo import models, api, fields, _
from odoo.exceptions import ValidationError
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from docstring_parser import parse
from odoo.exceptions import UserError

class Estimator(models.Model):
    _name = "supervised.learning.estimator"
    _description = "Supervised Learning Estimator"
    _order = "sequence"
    
    name = fields.Char('Name')
    dataset_id = fields.Many2one("supervised.learning.dataset", string="Dataset")
    model_id = fields.Many2one(related='dataset_id.model_id')
    dependent_variable_id = fields.Many2one(related='dataset_id.dependent_variable_id')
    independent_variable_ids = fields.Many2many(related='dataset_id.independent_variable_ids')
    sequence = fields.Integer(string="Sequence")
    column_ids = fields.Many2many('ir.model.fields', string="Columns", 
        domain="[('id', 'in', independent_variable_ids)]", 
        help="Columns to apply the estimator to. Leave empty if not applicable or if the estimator is applied to all columns.")
    unused_variable_ids = fields.Many2many('ir.model.fields', compute='_compute_unused_variable_ids')

    type = fields.Selection([
        ('ordinal_encoder', 'Ordinal Encoder'),
    ], string="Type")

    model = fields.Char(
        string='Model',
        related='model_id.model',
        precompute=True, store=True, readonly=True)

    def write(self, vals):
        all_in_draft = all(self.mapped(lambda record: record.dataset_id.state == 'draft'))

        if not all_in_draft:
            raise UserError(_('Writing to this field is not supported while not in state draft. Please reset to draft before continuing. Note that this means you will lose your training progress.'))

        return super().write(vals)

    @api.depends('column_ids', 'dataset_id.variable_ids')
    def _compute_unused_variable_ids(self):
        for record in self:
            record.unused_variable_ids = record.dataset_id.independent_variable_ids - record.column_ids

    def _get_estimator(self):
        self.ensure_one()
        raise ValidationError("No estimator found")