# Training on Azure ML

The same `train_model.py` that runs locally runs on Azure ML with **no code
changes** — Azure just provides the compute, scheduling, model versioning, and a
REST endpoint. This folder holds everything needed to submit the job.

## One-time setup

```bash
# 1. Install the CLI + ML extension, then sign in
brew install azure-cli          # macOS
az login
az extension add -n ml

# 2. Point the CLI at your workspace (create these in the Azure portal first)
az account set -s "<SUBSCRIPTION_ID>"
az configure --defaults group="<RESOURCE_GROUP>" workspace="<ML_WORKSPACE>"

# 3. Create a compute cluster that scales to zero when idle (pay only while training)
az ml compute create -n cpu-cluster --type AmlCompute \
  --min-instances 0 --max-instances 1 --size Standard_DS3_v2
```

## Train in the cloud

```bash
# Submit the job (opens the run in your browser with --web)
az ml job create -f azureml/train-job.yml --web
```

You'll see live logs and the metrics `train_model.py` prints. Because the
cluster scales to zero, cost is only the training minutes.

## Register the trained model (for deployment)

```bash
# Grab the run name from the previous command, then register its output model
az ml model create --name demand-forecast-model --type mlflow_model \
  --path "azureml://jobs/<RUN_NAME>/outputs/artifacts/models"
```

## Next steps (see ../TODO.md)

- Deploy a **Managed Online Endpoint** to serve `predict.py` as a REST API.
- Add an **Azure ML Pipeline** on a nightly schedule to retrain automatically.
- Create an **Azure OpenAI** resource for `assistant.py` (fill `../.env`).

> Everything here is push-button once you have an Azure subscription; there is
> nothing left to write on the code side.
