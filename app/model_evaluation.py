import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, df, target_column, test_size):
    from sklearn.model_selection import train_test_split

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Ensure all labels are consistent (convert to strings)
    y = y.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    y_pred = model.predict(X_test)
    y_pred = y_pred.astype(str)
    y_test = y_test.astype(str)

    # Print types for debugging
    print("y_test types:", y_test.dtypes)
    print("y_pred types:", type(y_pred))

    evaluation_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    confusion_matrix_plot = plot_confusion_matrix(y_test, y_pred)
    roc_curve_plot = plot_roc_curve(model, X_test, y_test)

    return evaluation_metrics, confusion_matrix_plot, roc_curve_plot

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    ax.set_xticks(range(len(cm)))
    ax.set_yticks(range(len(cm)))
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='red')
    return fig

def plot_roc_curve(model, X_test, y_test):
    if len(y_test.unique()) > 2:
        # Only plot ROC curve for binary classification
        return None

    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    y_test = y_test.astype(float)  # Convert y_test to numerical if it's not

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    return fig
