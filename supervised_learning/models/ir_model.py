from odoo import models, api, fields

class IRModel(models.Model):
    _inherit = 'ir.model'

    pipeline_ids = fields.One2many("supervised.learning.pipeline", "model_id", string="Pipelines")