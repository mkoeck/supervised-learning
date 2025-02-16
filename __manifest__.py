# -*- coding: utf-8 -*-
{
    'name': "Supervised Learning",
    'summary': "Machine Learning Pipeline Management",
    'description': """
This module provides tools for managing machine learning pipelines in Odoo.
Features include:
- Pipeline creation and management
- Support for PCA and SVC algorithms
- Step-by-step pipeline configuration
    """,
    'author': "My Company",
    'website': "https://www.yourcompany.com",
    'license': 'LGPL-3',
    'category': 'Technical',
    'version': '0.1',
    'depends': ['base', 'helpdesk'],
    'data': [
        'security/ir.model.access.csv',
        'views/pipeline_views.xml',
    ],
    'external_dependencies': {
        'python': ['scikit-learn'],
    },
    'demo': [
        'demo/demo.xml'
    ],

    'installable': True,
    'application': True,
    'post_init_hook': '_post_load_demo_data',
}
