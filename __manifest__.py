# -*- coding: utf-8 -*-
{
    'name': "Supervised Learning",
    'summary': "Machine Learning Pipeline Management",
    'description': """
This module provides tools for managing machine learning pipelines in Odoo.
    """,
    'author': "Michael Köck",
    'license': 'LGPL-3',
    'category': 'Technical',
    'version': '18.0.1.0.0',
    'depends': ['base', 'helpdesk'],
    'data': [
        'security/ir.model.access.csv',
        'views/pipeline_views.xml',
    ],
    'external_dependencies': {
        'python': ['scikit-learn', 'pandas'],
    },
    'demo': [
        'demo/demo.xml'
    ],

    'installable': True,
    'application': True,
    'post_init_hook': '_post_load_demo_data',
}
