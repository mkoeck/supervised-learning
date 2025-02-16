from odoo import models, api, fields
from sklearn.datasets import fetch_20newsgroups
import re
from odoo.fields import Command

def parse_email(text):
    email_pattern = re.compile(r"^From: (.+?) <(.+?)>$", re.MULTILINE)
    subject_pattern = re.compile(r"^Subject: (.+)$", re.MULTILINE)
    organization_pattern = re.compile(r"^Organization: (.+)$", re.MULTILINE)
    body_pattern = re.compile(r"\n\n(.+)$", re.DOTALL)
    
    email_match = email_pattern.search(text)
    subject_match = subject_pattern.search(text)
    organization_match = organization_pattern.search(text)
    body_match = body_pattern.search(text)
    
    email_info = {
        "sender_name": email_match.group(1) if email_match else None,
        "sender_email": email_match.group(2) if email_match else None,
        "subject": subject_match.group(1) if subject_match else None,
        "organization": organization_match.group(1) if organization_match else None,
        "body": body_match.group(1).strip() if body_match else None,
    }
    
    return email_info

class HelpdeskTicket(models.Model):
    _inherit = 'helpdesk.ticket'

    def _post_load_demo_data(self):
        data = fetch_20newsgroups(
            subset="all",
            random_state=42,
        )
        records = []

        tags = [{
                'id': name.replace('.', '_'),
                'name': name,
            } for name in data.target_names
        ]

        tag_ids = self.env['helpdesk.tag'].with_context(_import_current_module='supervised_learning', noupdate=True).create(tags)

        for i, (record, target) in enumerate(zip(data.data, data.target)):
            target_name = data.target_names[target]
            parsed_email = parse_email(record)
            records.append({
                'id': f'helpdesk_ticket_{i}',
                'name': parsed_email['subject'],
                'description': parsed_email['body'],
                'tag_ids': [Command.set([self.env.ref(f'supervised_learning.{target_name.replace(".", "_")}').id])],
                'stage_id': self.env.ref('helpdesk.stage_new').id,
                'team_id': self.env.ref('helpdesk.helpdesk_team1').id,
            })

        tickets = self.env['helpdesk.ticket'].with_context(_import_current_module='supervised_learning', noupdate=True).create(records)