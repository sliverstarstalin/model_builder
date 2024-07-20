# app/routes.py
import os
from flask import Blueprint, render_template, url_for, flash, redirect, request, send_from_directory
from werkzeug.utils import secure_filename
from app import db, app
from app.forms import RegistrationForm, LoginForm, UploadForm, EmptyForm, DataCleaningForm,ModelBuildingForm
from app.models import User
from flask_login import login_user, current_user, logout_user, login_required
from flask_bcrypt import Bcrypt
from app.data_preprocessing import load_data, handle_missing_values, normalize_data, standardize_data, remove_outliers, encode_categorical, log_and_report, compute_data_metrics, handle_duplicates, save_preprocessing_pipeline, load_preprocessing_pipeline
from app.model_building import build_supervised_model, build_unsupervised_model, save_model, load_model
from app.model_evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve

bcrypt = Bcrypt(app)

bp = Blueprint('main', __name__)
UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route("/")
@bp.route("/home")
def home():
    return render_template('index.html')

@bp.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('main.login'))
    return render_template('signup.html', title='Register', form=form)

@bp.route("/login", methods=['GET', 'POST'])
def login():
    print("Login route accessed")
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    form = LoginForm()
    if form.validate_on_submit():
        print("Form validated")
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('main.home'))
        else:
            print("Invalid email or password")
            flash('Login Unsuccessful. Please check email and password', 'danger')
    else:
        print("Form did not validate")
    return render_template('login.html', title='Login', form=form)

@bp.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('main.home'))

@bp.route("/upload", methods=['GET', 'POST'])
@login_required
def upload():
    form = UploadForm()
    file_path = None

    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        flash('File uploaded successfully', 'success')
        return redirect(url_for('main.upload', file_path=filename))

    if 'file_path' in request.args:
        file_path = request.args.get('file_path')

    return render_template('upload.html', title='Upload File', form=form, file_path=file_path)


@bp.route("/clean_data/<path:file_path>", methods=['GET', 'POST'])
@login_required
def clean_data(file_path):
    print(f"Accessing clean_data with file_path: {file_path}")
    full_file_path = os.path.join(UPLOAD_FOLDER, file_path)
    print(f"Full file path: {full_file_path}")
    df = load_data(full_file_path)
    form = DataCleaningForm()
    metrics = compute_data_metrics(df)
    if request.method == 'POST':
        print("Handling POST request for clean_data")
        print("Form data:", request.form)
        if form.validate_on_submit():
            if form.remove_duplicates.data:
                df = handle_duplicates(df, method=form.duplicate_handling_strategy.data)
            if form.remove_outliers.data:
                df = remove_outliers(df)
            strategy = form.missing_values_strategy.data
            df = handle_missing_values(df, strategy)
            if form.normalize.data:
                df = normalize_data(df)
            if form.standardize.data:
                df = standardize_data(df)
            if form.encode_categorical.data:
                df = encode_categorical(df, form.encoding_strategy.data)
            clean_file_path = os.path.join(UPLOAD_FOLDER, 'cleaned_' + os.path.basename(file_path))
            print(f"Saving cleaned data to: {clean_file_path}")
            df.to_csv(clean_file_path, index=False)
            log_and_report(df, report_file=os.path.join(UPLOAD_FOLDER, 'report_' + os.path.basename(file_path).replace('.', '_') + '.txt'))
            flash('Data cleaned successfully', 'success')
            return send_from_directory(directory=UPLOAD_FOLDER, path='cleaned_' + os.path.basename(file_path), as_attachment=True)
    return render_template('clean_data.html', title='Clean Data', columns=df.columns, form=form, file_path=file_path, metrics=metrics)

# app/routes.py


@bp.route("/build_model/<path:file_path>", methods=['GET', 'POST'])
@login_required
def build_model(file_path):
    full_file_path = os.path.join(UPLOAD_FOLDER, file_path)
    df = load_data(full_file_path)
    form = ModelBuildingForm()
    model_performance = None

    if request.method == 'POST' and form.validate_on_submit():
        model_type = form.model_type.data
        target_column = form.target_column.data
        test_size = float(form.test_size.data)
        
        if model_type in ['linear_regression', 'logistic_regression', 'decision_tree', 'random_forest']:
            model, performance = build_supervised_model(df, target_column, model_type, test_size)
            print("model built",model,performance)
        elif model_type == 'kmeans':
            model, labels = build_unsupervised_model(df, model_type)
            performance = None  # KMeans does not have a single performance metric

        model_file_path = os.path.join(UPLOAD_FOLDER, f"{model_type}_model.pkl")
        save_model(model, model_file_path)
        print("model saved",model_file_path)
        flash(f'Model built and saved successfully with performance: {performance}', 'success')
        return send_from_directory(directory=UPLOAD_FOLDER, path=os.path.basename(model_file_path), as_attachment=True)

    return render_template('model_building.html', title='Build Model', form=form, columns=df.columns, model_performance=model_performance)


@bp.route("/evaluate_model/<path:file_path>", methods=['GET', 'POST'])
@login_required
def evaluate_model_route(file_path):
    full_file_path = os.path.join(UPLOAD_FOLDER, file_path)
    df = load_data(full_file_path)
    form = ModelBuildingForm()  # Reusing the form for simplicity
    evaluation_metrics = None
    confusion_matrix_plot = None
    roc_curve_plot = None

    if request.method == 'POST' and form.validate_on_submit():
        model_type = form.model_type.data
        target_column = form.target_column.data
        test_size = float(form.test_size.data)
        
        model_file_path = os.path.join(UPLOAD_FOLDER, f"{model_type}_model.pkl")
        model = load_model(model_file_path)

        evaluation_metrics, confusion_matrix_plot, roc_curve_plot = evaluate_model(model, df, target_column, test_size)

        # Save plots and return their file paths
        confusion_matrix_path = os.path.join(UPLOAD_FOLDER, 'confusion_matrix.png')
        confusion_matrix_plot.savefig(confusion_matrix_path)

        if roc_curve_plot:
            roc_curve_path = os.path.join(UPLOAD_FOLDER, 'roc_curve.png')
            roc_curve_plot.savefig(roc_curve_path)
            return render_template('evaluate_model.html', title='Evaluate Model', form=form, evaluation_metrics=evaluation_metrics, confusion_matrix_path=os.path.basename(confusion_matrix_path), roc_curve_path=os.path.basename(roc_curve_path))
        else:
            return render_template('evaluate_model.html', title='Evaluate Model', form=form, evaluation_metrics=evaluation_metrics, confusion_matrix_path=os.path.basename(confusion_matrix_path), roc_curve_path=None)

    return render_template('evaluate_model.html', title='Evaluate Model', form=form, evaluation_metrics=evaluation_metrics)
