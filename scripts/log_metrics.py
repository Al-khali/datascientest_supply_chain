import os
import mlflow
import numpy as np
import matplotlib.pyplot as plt  # Add this import
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
        
        # Generate and save the confusion matrix
        confusion_matrix = np.array([[150, 25], [15, 160]])
        plt.imshow(confusion_matrix, cmap='Blues')
        plt.colorbar()
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig("confusion_matrix.png")  # Save the file
        plt.close()
        
        # Log the confusion matrix artifact
        mlflow.log_artifact("confusion_matrix.png")
        
        print("Logged metrics to MLflow successfully")

if __name__ == "__main__":
    main()
