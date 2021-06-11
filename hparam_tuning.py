"""
Run Hyperparameter Tuning job on GCP Vertex AI. 
Created on: 11 June 2021. 
"""

from google.cloud import aiplatform


def create_hyperparameter_tuning_job_python_package(
    project: str = "fifthproj",
    display_name: str = "ariel ml hyperparameter tuning",
    executor_image_uri: str = "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-4:latest",
    package_uri: str = "gs://trainingbucket3113/",
    python_module: str = "train_second",
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.JobServiceClient(client_options=client_options)

    metric = {
        "metric_id": "ariel_score",
        "goal": aiplatform.gapic.StudySpec.MetricSpec.GoalType.MAXIMIZE,
    }

    conditional_param_H1 = {
        "parameter_spec": {
            "parameter_id": "H1",
            "discrete_value_spec": {"values": [4, 8, 16, 32, 64, 128, 256, 512, 1024]},
            "scale_type": aiplatform.gapic.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
        },
        "parent_discrete_values": {"values": [10, 25, 50, 100]}
    }

    conditional_param_H2 = {
        "parameter_spec": {
            "parameter_id": "H2",
            "discrete_value_spec": {"values": [64, 128, 256, 512, 1024]},
            "scale_type": aiplatform.gapic.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
        },
        "parent_discrete_values": {"values": [10, 25, 50, 100]}
    }


    conditional_param_H3 = {
        "parameter_spec": {
            "parameter_id": "H3",
            "discrete_value_spec": {"values": [4, 8, 16, 32, 64, 128, 256, 512, 1024]},
            "scale_type": aiplatform.gapic.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
        },
        "parent_discrete_values": {"values": [10, 25, 50, 100]}
    }

    conditional_param_D1 = {
        "parameter_spec": {
            "parameter_id": "D1",
            "double_value_spec": {"min_value": 0.01, "max_value": 0.5},
            "scale_type": aiplatform.gapic.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
        },
        "parent_discrete_values": {"values": [10, 25, 50, 100]}
    }

    conditional_param_mean = {
        "parameter_spec": {
            "parameter_id": "mean",
            "discrete_value_spec": {"values": [0., 1.]},
            "scale_type": aiplatform.gapic.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
        },
        "parent_discrete_values": {"values": [10, 25, 50, 100]}
    }

    conditional_param_std = {
        "parameter_spec": {
            "parameter_id": "std",
            "double_value_spec": {"min_value": 0.005, "max_value": 0.5},
            "scale_type": aiplatform.gapic.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
        },
        "parent_discrete_values": {"values": [10, 25, 50, 100]}
    }

    conditional_param_lr = {
        "parameter_spec": {
            "parameter_id": "lr",
            "discrete_value_spec": {"values": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]},
            "scale_type": aiplatform.gapic.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
        },
        "parent_discrete_values": {"values": [10, 25, 50, 100]}
    }

    parameter = {
        "parameter_id": "batch_size",
        "discrete_value_spec": {"values": [10, 25, 50, 100]},
        "scale_type": aiplatform.gapic.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
        "conditional_parameter_specs": [
            conditional_param_H1, 
            conditional_param_H2,
            conditional_param_H3,
            conditional_param_D1,
            conditional_param_mean,
            conditional_param_std,
            conditional_param_lr,
        ],
    }

    # Trial job spec
    machine_spec = {
        "machine_type": "e2-standard-4",
    }
    worker_pool_spec = {
        "machine_spec": machine_spec,
        "replica_count": 1, 
        "python_package_spec": {
            "executor_image_uri": executor_image_uri,
            "package_uris": [package_uri],
            "python_module": python_module,
            "args": [],
        }
    }

    # hparam tuning job
    hyperparameter_tuning_job = {
        "display_name": display_name,
        "max_trial_count": 2, 
        "parallel_trial_count": 2,
        "study_spec": {
            "metrics": [metric],
            "parameters": [parameter],
        },
        "trial_job_spec": {"worker_pool_specs": [worker_pool_spec]},
    }

    parent = f"projects/{project}/locations/{location}"
    response = client.create_hyperparameter_tuning_job(
        parent=parent, hyperparameter_tuning_job=hyperparameter_tuning_job
    )
    print(f"response:", response)

