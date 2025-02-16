from odoo import models, api, fields, _
from odoo.exceptions import UserError
from odoo.tools.safe_eval import safe_eval
from sklearn.pipeline import Pipeline
from typing import Optional
import pandas as pd
from sklearn.utils import estimator_html_repr
import pickle
import base64
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils._param_validation import InvalidParameterError 
from sklearn.preprocessing import MultiLabelBinarizer

# TODO:
# introduce training via ir.cron, with regular retraining

class Dataset(models.Model):
    _name = "supervised.learning.dataset"
    _description = "Supervised Learning Dataset"
    
    name = fields.Char(string="Name")
    model_id = fields.Many2one("ir.model", string="Model")
    model_name = fields.Char(related='model_id.model')
    domain = fields.Char()
    state = fields.Selection([
        ('draft', 'Draft'),
        ('trained', 'Trained'),
    ], string="State", default='draft', required=True)
    independent_variable_ids = fields.Many2many(
        'ir.model.fields',
        domain="[('model_id', '=', model_id)]",
        string="Independent Variables"
    )
    dependent_variable_id = fields.Many2one(
        'ir.model.fields',
        domain="[('model_id', '=', model_id)]",
        string="Dependent Variable"
    )

    variable_ids = fields.Many2many('ir.model.fields', compute='_compute_variable_ids')
    estimator_ids = fields.One2many("supervised.learning.estimator", "dataset_id", string="Steps")

    # performance metrics
    precision = fields.Float(readonly=True)
    recall = fields.Float(readonly=True)
    f1_score = fields.Float(readonly=True)

    pipeline = fields.Binary(compute='_compute_pipeline', store=True, attachment=True)
    pipeline_html = fields.Html(compute='_compute_pipeline_html', sanitize=False)

    dependent_variable_preprocessor = fields.Binary(compute='_compute_dependent_variable_preprocessor', store=True, attachment=True)

    is_classification = fields.Boolean(compute='_compute_is_classification')
    is_multilabel = fields.Boolean(compute='_compute_is_classification')

    action_id = fields.Many2one('ir.actions.server')

    def write(self, vals):
        always_allowed_fields = {'state', 'action_id'}
        fields_to_write = set(vals.keys())

        only_writing_allowed_fields = fields_to_write.issubset(always_allowed_fields)
        all_in_draft = all(self.mapped(lambda record: record.state == 'draft'))

        if not all_in_draft and not only_writing_allowed_fields:
            raise UserError(_('Writing to this field is not supported while not in state draft. Please reset to draft before continuing. Note that this means you will lose your training progress.'))

        return super().write(vals)

    def action_reset_to_draft(self):
        self.write({'state': 'draft'})

    def create_action(self):
        for record in self:

            code = f"""
dataset = env['supervised.learning.dataset'].browse({record.id})
predictions = dataset.predict(records)
for i, rec in enumerate(records):
    rec[dataset.dependent_variable_id.name] = predictions[i]
            """

            self.action_id = self.env['ir.actions.server'].create({
                'state': 'code',
                'model_id': record.model_id.id,
                'binding_model_id': record.model_id.id,
                'binding_type': 'action',
                'code': code,
                'name': _('Predict %s', record.dependent_variable_id.field_description)
            })

    def unlink_action(self):
        for record in self:
            if record.action_id:
                record.action_id.unlink()

    def predict(self, records):
        """
        Method that takes a recordset and returns a prediction for each record.
        """
        self.ensure_one()
        if records._name != self.model_id.model:
            raise UserError(_("Expected record of model %s, got %s", self.model_id.model, records._name))

        pipeline = self._get_pipeline()

        if self.state == 'draft' or not pipeline:
            raise UserError(_("Can only make predictions on trained datasets"))

        dataset = self._get_dataset(domain=[('id', 'in', records.ids)])

        X = dataset[self.independent_variable_ids.mapped('name')]
        y_pred = pipeline.predict(X)

        dependent_variable_preprocessor = self._get_dependent_variable_preprocessor()
        if dependent_variable_preprocessor:
            y_pred = dependent_variable_preprocessor.inverse_transform(y_pred)
            transformed_y_pred = []
            for prediction in y_pred:
                if isinstance(prediction, tuple):
                    transformed_y_pred.append(tuple(x.item() for x in prediction)) # convert numpy scalars to python scalars
                else:
                    raise ValueError("Unexpected data type %s", type(prediction)) # TODO: are there other data types that need to be handled?
            y_pred = transformed_y_pred
        return y_pred

    @api.depends('independent_variable_ids', 'dependent_variable_id')
    def _compute_variable_ids(self):
        for record in self:
            record.variable_ids = record.independent_variable_ids + record.dependent_variable_id

    @api.depends('dependent_variable_id')
    def _compute_is_classification(self):
        for record in self:
            record.is_classification = record.dependent_variable_id.ttype in ['many2one', 'many2many', 'one2many', 'selection', 'boolean', 'char', 'text', 'html', 'reference', 'many2one_reference']
            record.is_multilabel = record.dependent_variable_id.ttype in ['many2many', 'one2many']

    def _get_dataset(self, **kwargs) -> pd.DataFrame:
        """
        Creates a pandas dataframe according to the specifications. Note that by using kwargs,
        additional parameters can be passed to search_read. This is especially useful, to pass a custom domain,
        to create a dataset for specific records, instead of all of them. An example can be found in the predict method.
        """
        self.ensure_one()
        model = self.env[self.model_id.model]
        fields = self.variable_ids.mapped('name')
        
        domain = kwargs.pop('domain', [])
        data = model.search_read(domain, fields, **kwargs)
        df = pd.DataFrame(data)
        df.index = df.pop('id')
        return self._preprocess_dataset(df)

    def _preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method does some preprocessing on the dataset that should be done before applying the pipeline.
        Note that most of the preprocessing should be done in the pipeline itself, but some of it is done here because it can't be done in the pipeline.
        """
        df = self._replace_null_values(df)
        return df
    
    def _replace_null_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method preprocesses a dataset created by _get_dataset to get from an 'odoo' dataset to a 'sklearn' dataset.
        Odoo largely uses False to represent missing values, while sklearn requires NA.
        """
        for variable in self.variable_ids:
            if variable.ttype != 'boolean':
                df[variable.name] = df[variable.name].replace({False: pd.NA})
            if variable.ttype in ['text', 'char', 'html']:
                df[variable.name] = df[variable.name].replace({'': pd.NA})
        return df

    @api.depends('pipeline')
    def _compute_pipeline_html(self):
        for record in self:
            html = estimator_html_repr(record._get_pipeline())
            record.pipeline_html = html

    @api.depends('estimator_ids', 'estimator_ids.sequence')
    def _compute_pipeline(self):
        for record in self:
            pipeline = [(estimator.name, estimator._get_estimator()) for estimator in record.estimator_ids]
            if not pipeline:
                record.pipeline = False
            else:
                pipeline = Pipeline(pipeline)
                record._set_pipeline(pipeline)

    def _set_pipeline(self, pipeline):
        self.ensure_one()
        pickled_pipeline = pickle.dumps(pipeline)
        self.pipeline = base64.b64encode(pickled_pipeline)

    def _get_pipeline(self) -> Optional[Pipeline]:
        self.ensure_one()
        if not self.pipeline:
            return None
        return pickle.loads(base64.b64decode(self.with_context(bin_size=False, bin_size_pipeline=False).pipeline))

    @api.depends('dependent_variable_id', 'independent_variable_ids')
    def _compute_dependent_variable_preprocessor(self):
        for record in self:
            if record.is_multilabel:
                data = record._get_dataset()
                preprocessor = MultiLabelBinarizer()
                preprocessor.fit(data[record.dependent_variable_id.name])
                record.dependent_variable_preprocessor = base64.b64encode(pickle.dumps(preprocessor))
            else:
                record.dependent_variable_preprocessor = False

    def _get_dependent_variable_preprocessor(self):
        self.ensure_one()
        if not self.dependent_variable_preprocessor:
            return None
        return pickle.loads(base64.b64decode(self.dependent_variable_preprocessor))

    def _set_performance_metrics(self, y_test, y_pred):
        self.ensure_one()
        if self.is_classification:
            self.f1_score = f1_score(y_test, y_pred, average='weighted')
            self.precision = precision_score(y_test, y_pred, average='weighted')
            self.recall = recall_score(y_test, y_pred, average='weighted')

    def action_run_pipeline(self):
        for record in self:
            pipeline = record._get_pipeline()
            data = record._get_dataset(domain=safe_eval(record.domain))
            
            # TODO: during training, we should not have any missing values in the dependent variable
            # given that the pipeline can not be applied to the dependent variable, we need to deal with missing values here
            # dropping these rows is a simple solution, but not the only one
            # we could also impute the missing values, or use a separate label for missing values
            data = data.dropna(subset=[self.dependent_variable_id.name])

            if not pipeline:
                raise UserError(_("No pipeline defined"))

            X = data[record.independent_variable_ids.mapped('name')]
            y = data[record.dependent_variable_id.name]

            dependent_variable_preprocessor = record._get_dependent_variable_preprocessor()
            if dependent_variable_preprocessor:
                y = dependent_variable_preprocessor.transform(y)

            # TODO: introduce better testing like cross-validation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            try:
                pipeline.fit(X_train, y_train)
            except InvalidParameterError as e:
                raise UserError(str(e))

            record._set_pipeline(pipeline)

            y_pred = pipeline.predict(X_test)
            record._set_performance_metrics(y_test, y_pred)

        self.write({'state': 'trained'})

        return {
            'type': 'ir.actions.client',
            'tag': 'reload',
        }