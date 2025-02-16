from odoo import api, exceptions, models, tools, _

class BaseModel(models.AbstractModel):
    _inherit = 'base'