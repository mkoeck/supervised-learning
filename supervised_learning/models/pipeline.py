from odoo import models, api, fields, _
from odoo.exceptions import UserError, ValidationError
from odoo.tools.safe_eval import safe_eval
from sklearn.pipeline import Pipeline as SklearnPipeline
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

class Pipeline(models.Model):
    _name = "supervised.learning.pipeline"
    _description = "Supervised Learning Pipeline"
    
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
        relation='sl_pipeline_independent_var_rel',
        domain="[('model_id', '=', model_id)]",
        string="Independent Variables"
    )
    dependent_variable_id = fields.Many2one(
        'ir.model.fields',
        domain="[('model_id', '=', model_id)]",
        string="Dependent Variable"
    )

    is_grouped = fields.Boolean('Group Data', default=False)

    groupby_field_ids = fields.Many2many(
        'ir.model.fields',
        relation='sl_pipeline_groupby_fields_rel',
        domain="[('model_id', '=', model_id)]",
        string="Grouping Variables"
    )

    date_group_frequency = fields.Selection([
        ('year_number', 'Year'),
        ('quarter_number', 'Quarter'),
        ('month_number', 'Month'),
        ('iso_week_number', 'Calendar Week'),
        ('day_of_year', 'Day of Year'),
        ('day_of_month', 'Day of Month'),
        ('day_of_week', 'Day of Week'),
        ('hour_number', 'Hour Number'),
        ('minute_number', 'Minute Number'),
        ('second_number', 'Second Number')
    ], default='day_of_year')

    aggregation_field_ids = fields.One2many(
        'supervised.learning.aggregation.field',
        'pipeline_id',
    )

    variable_ids = fields.Many2many('ir.model.fields', compute='_compute_variable_ids')
    estimator_ids = fields.One2many("supervised.learning.estimator", "pipeline_id", string="Steps")

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

    @api.constrains('is_grouped', 'variable_ids')
    def _check_is_grouped_valid_variables(self):
        for record in self:
            if not record.is_grouped:
                continue
            
            for variable in record.variable_ids:
                is_aggregation_variable = variable in record.aggregation_field_ids.mapped('field_id')
                is_grouping_variable = variable in record.groupby_field_ids
                if not is_aggregation_variable and not is_grouping_variable:
                    raise ValidationError(_('When using grouping, each variable must either be an aggregation or grouping variable. "%s" is neither', variable.name))

    def action_reset_to_draft(self):
        self.write({'state': 'draft'})

    def action_export_dataset(self):
        self.ensure_one()
        return {
            "type": "ir.actions.act_url",
            "url": f"/supervised_learning/export/dataset/{self.id}",
            "target": "download",
        }

    def create_action(self):
        for record in self:

            code = f"""
pipeline = env['supervised.learning.pipeline'].browse({record.id})
predictions = pipeline.predict(records)
for i, rec in enumerate(records):
    rec[pipeline.dependent_variable_id.name] = predictions[i]
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
            raise UserError(_("Can only make predictions on trained pipelines"))

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
        Creates a pandas DataFrame according to specifications.
        If `self.is_grouped` is False, uses search_read logic;
        otherwise uses read_group logic.
        """
        self.ensure_one()
        domain = kwargs.pop("domain", [])

        if self.is_grouped:
            df = self._get_dataset_grouped(domain, **kwargs)
        else:
            df = self._get_dataset_unaggregated(domain, **kwargs)

        # Preprocess before returning
        return self._preprocess_pipeline(df)

    def _get_dataset_unaggregated(self, domain, **kwargs) -> pd.DataFrame:
        """Return a DataFrame via a simple search_read."""
        model = self.env[self.model_id.model]
        fields = self.variable_ids.mapped("name")
        data = model.search_read(domain, fields, **kwargs)
        return pd.DataFrame(data, columns=pd.Series(fields))


    def _get_dataset_grouped(self, domain, **kwargs) -> pd.DataFrame:
        """Return a DataFrame via read_group for grouped logic."""
        model = self.env[self.model_id.model]
        freq = self.date_group_frequency
        # Helper to detect date/datetime fields
        is_date_field = lambda f: f.ttype in ["date", "datetime"]

        # Identify fields that should be grouped by date/datetime frequency
        grouped_date_fields = {
            field.name for field in self.groupby_field_ids if is_date_field(field)
        }

        # Build groupby fields
        groupby_fields = [
            f"{field.name}:{freq}" if field.name in grouped_date_fields else field.name
            for field in self.groupby_field_ids
        ]
        read_group_args = self.aggregation_field_ids.get_read_group_argument()

        # Fetch data
        data = model.read_group(domain, read_group_args, groupby_fields, lazy=False, **kwargs)

        # Build columns for the final DataFrame
        columns = [
            f"{field.name}:{freq}" if field.name in grouped_date_fields else field.name
            for field in self.variable_ids
        ]
        # Rename map to remove the ":freq" suffix for date/datetime fields
        rename_map = {
            f"{field.name}:{freq}": field.name
            for field in self.variable_ids
            if field.name in grouped_date_fields
        }

        return pd.DataFrame(data, columns=pd.Series(columns)).rename(columns=rename_map, inplace=False)

    def _preprocess_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method does some preprocessing on the pipeline that should be done before applying the pipeline.
        Note that most of the preprocessing should be done in the pipeline itself, but some of it is done here because it can't be done in the pipeline.
        """
        df = self._preprocess_columns(df)
        return df
    
    def _preprocess_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        self.ensure_one()
        for variable in self.variable_ids:
            variable_name = variable.name
            if variable.ttype == 'many2one':
                df = self._preprocess_column_many2one(df, variable_name)
            if variable.ttype != 'boolean':
                df = self._preprocess_column_not_bool(df, variable_name)
            if variable.ttype in ['text', 'char', 'html']:
                df = self._preprocess_column_str(df, variable_name)
        return df

    @api.model
    def _preprocess_column_many2one(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        df[column_name] = df.loc[:,column_name].str[0] # many2one fields are tuples in the form of (id, name)
        return df

    @api.model
    def _preprocess_column_not_bool(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        df[column_name] = df.loc[:,column_name].replace({False: pd.NA})
        return df

    @api.model
    def _preprocess_column_str(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        df[column_name] = df.loc[:,column_name].replace({'': pd.NA})
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
                pipeline = SklearnPipeline(pipeline)
                record._set_pipeline(pipeline)

    def _set_pipeline(self, pipeline):
        self.ensure_one()
        pickled_pipeline = pickle.dumps(pipeline)
        self.pipeline = base64.b64encode(pickled_pipeline)

    def _get_pipeline(self) -> Optional[SklearnPipeline]:
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
            domain = record.domain or '[]'
            data = record._get_dataset(domain=safe_eval(domain))
            
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