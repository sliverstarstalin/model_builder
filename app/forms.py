# app/forms.py
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, SelectField, FileField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from app.models import User

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('That username is taken. Please choose a different one.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('That email is already in use. Please choose a different one.')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

class UploadForm(FlaskForm):
    file = FileField('Upload File', validators=[DataRequired()])
    submit = SubmitField('Upload')

class DataCleaningForm(FlaskForm):
    missing_values_strategy = SelectField('Missing Values Strategy', choices=[('mean', 'Mean'), ('median', 'Median'), ('mode', 'Mode'), ('drop', 'Drop Rows')], default='mean', validators=[DataRequired()])
    duplicate_handling_strategy = SelectField('Duplicate Handling Strategy', choices=[('drop', 'Drop All Duplicates'), ('keep_first', 'Keep First Duplicate'), ('keep_last', 'Keep Last Duplicate')], default='drop', validators=[DataRequired()])
    remove_duplicates = BooleanField('Remove Duplicates')
    remove_outliers = BooleanField('Remove Outliers')
    normalize = BooleanField('Normalize Data')
    standardize = BooleanField('Standardize Data')
    encode_categorical = BooleanField('Encode Categorical Data')
    encoding_strategy = SelectField('Encoding Strategy', choices=[('onehot', 'One Hot Encoding'), ('label', 'Label Encoding')], default='onehot', validators=[DataRequired()])
    submit = SubmitField('Clean Data')

class EmptyForm(FlaskForm):
    submit = SubmitField('Submit')
# app/forms.py
class ModelBuildingForm(FlaskForm):
    model_type = SelectField('Model Type', choices=[('linear_regression', 'Linear Regression'), ('logistic_regression', 'Logistic Regression'), ('decision_tree', 'Decision Tree'), ('random_forest', 'Random Forest'), ('kmeans', 'KMeans')], default='linear_regression', validators=[DataRequired()])
    target_column = StringField('Target Column', validators=[DataRequired()])
    test_size = StringField('Test Size', default='0.2', validators=[DataRequired()])
    submit = SubmitField('Build Model')
