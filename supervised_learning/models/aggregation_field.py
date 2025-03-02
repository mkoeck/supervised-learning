from odoo import models, api, fields
from typing import List

class AggregationField(models.Model):
    _name = 'supervised.learning.aggregation.field'

    valid_field_ids = fields.Many2many(related='pipeline_id.variable_ids')

    pipeline_id = fields.Many2one('supervised.learning.pipeline', required=True)
    field_id = fields.Many2one('ir.model.fields', required=True, ondelete='cascade', domain="[('id', 'in', valid_field_ids)]")
    aggregation_function = fields.Selection([
        ('sum', 'Sum'),
        ('count', 'Count'),
        ('avg', 'Average'),
        ('min', 'Minimum'),
        ('max', 'Maximum')
    ], string="Aggregation Function", default='sum', required=True)

    def get_read_group_argument(self) -> List[str]:
        """
        Helper function that formats the recordset in such a way that it can be passed to 'read_group'
        """
        return [f'{r.field_id.name}:{r.aggregation_function}' for r in self]