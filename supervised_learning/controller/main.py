# -*- coding: utf-8 -*-
from odoo import http
from odoo.http import request
from odoo.tools.safe_eval import safe_eval

class PipelineController(http.Controller):

    @http.route(['/supervised_learning/export/dataset/<int:pipeline_id>'],
                type='http',
                auth='user')
    def export_dataset(self, pipeline_id, **kwargs):
        """
        Download the dataset for the specified pipeline as a CSV file.
        """
        # Locate the pipeline
        pipeline = request.env['supervised.learning.pipeline'].browse(pipeline_id)
        if not pipeline.exists():
            return request.not_found()

        # Get the dataset as a pandas DataFrame
        dataset = pipeline._get_dataset()
        # Convert DataFrame to CSV
        csv_content = dataset.to_csv(index=False)

        # Return as an attachment download
        filename = f"pipeline_{pipeline_id}_dataset.csv"
        return request.make_response(
            csv_content,
            headers=[
                ('Content-Type', 'text/csv'),
                ('Content-Disposition', f'attachment; filename="{filename}"')
            ]
        )
