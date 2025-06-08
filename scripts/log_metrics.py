import os
import mlflow
import numpy as np
from datetime import datetime

def main():
    # Connect to MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("ci-pipeline")
    
    # Log sample metrics (in production, these would come from actual model evaluation)
    with mlflow.start_run(run_name=f"ci-run-{datetime.now().isoformat()}"):
        mlflow.log_param("model_type", "sentiment_analysis")
        mlflow.log_metric("accuracy", 0.92)
        mlflow.log_metric("f1_score", 0.89)
        mlflow.log_metric("precision", 0.91)
        mlflow.log_metric("recall", 0.90)
        mlflow.log_metric("roc_auc", 0.96)
        
        # Log sample confusion matrix
        confusion_matrix = np.array([[150, 25], [15, 160]])
        mlflow.log_artifact("confusion_matrix.png")
        
        print("Logged metrics to MLflow successfully")

if __name__ == "__main__":
    main()
