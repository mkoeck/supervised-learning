from odoo import models, api, fields

class IRModel(models.Model):
    _inherit = 'ir.model'

    dataset_ids = fields.One2many("supervised.learning.dataset", "model_id", string="Datasets")