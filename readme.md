## Model Deployment using BentoML & AWS Fargate

This example shows how to deploy a machine learning model to AWS Fargate using BentoML. The example uses the [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset) dataset to train a scikit-learn model, and uses BentoML to package the model as a REST API server. The REST API server is then deployed to AWS Fargate using BentoML's CLI tool.

[Train](codes\train_model.py) code comes from [this example](https://github.com/silverstone1903/mlops/blob/master/CML/train_model.py). For the simplictity, model only uses the 8 selected columns for training and prediction. I didn't want to make the example too complicated and keep the focus on the deployment part.

### Usage

To run this example, you will need to have Docker installed and running on your local machine. You will also need to have an AWS account and have the AWS CLI installed and configured on your local machine.

First run the following code to save the model to a BentoService bundle:

```bash
python codes/train_model.py
```

Then run the following code to start the REST API server:

```bash
cd codes
bentoml serve --production predict.py:svc
```

Then build the image:

```bash
bentoml containerize hr_attrition_model:latest
```

It will show the container image ID after the build is complete. You can use the image ID to run the container locally:

```bash
docker run -it --rm -p 3000:3000 hr_attrition_model:cztevllezwqizvc5 serve --production
``` 

If everything works, you should be able to send a request to the REST API server using curl (Postman is also a good option):

```bash
curl -X 'POST' \
  'localhost:3000/classify' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "BusinessTravel": 2,
    "DailyRate": 279,
    "Department": 2,
    "DistanceFromHome": 1,
    "Education": 2,
    "EducationField": 4,
    "EnvironmentSatisfaction": 1,
    "Gender": 1
}'
```	

### Fargate Deployment

#### Push the image to ECR

First install the awscli if you don't have. Also, make sure you have the AWS credentials configured.

```bash	
pip install awscli
```

Create a registry in ECR
```bash	
aws ecr create-repository --repository-name repo-name
```

Authentication for ECR

```bash	
aws ecr get-login-password --region eu-x | docker login --username AWS --password-stdin xxx.dkr.ecr.eu-x.amazonaws.com
```

Tag and Push
```bash
docker tag hr_attrition_model:cztevllezwqizvc5 xxx.dkr.ecr.eu-x.amazonaws.com/hr_attrition_model:latest 
docker push xxx.dkr.ecr.eu-x.amazonaws.com/hr_attrition_model:latest 
``` 

#### Fargate Cluster
1. Create a cluster (networking only)
2. Create a task definition (task role, container definition, memory, cpu, etc.)
3. Run the task!

If everything went well, you should be able to see the task running in the ECS console. Then you can check the public IP of the task and send a request to the REST API server.

You can check the [Deploying a Docker container with ECS and Fargate](https://towardsdatascience.com/deploying-a-docker-container-with-ecs-and-fargate-7b0cbc9cd608) for more details.

#### Test the API

```python
import requests
import pprint
pp = pprint.PrettyPrinter(indent=4)

url = "https://your-fargate-public-ip:3000/classify"
data = {
    "BusinessTravel": 1,
    "DailyRate": 278,
    "Department": 1,
    "DistanceFromHome": 7,
    "Education": 3,
    "EducationField": 4,
    "EnvironmentSatisfaction": 1,
    "Gender": 0
}
result = requests.post(url, json=data).json()
pp.pprint(result)
# {'Attrition': 'Yes'}
```

### Resources
1. [BentoML](https://docs.bentoml.org/en/latest/concepts/model.html)
2. [Deploying a Docker container with ECS and Fargate](https://towardsdatascience.com/deploying-a-docker-container-with-ecs-and-fargate-7b0cbc9cd608)
3. [Production-Ready Machine Learning (Bento ML)](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/07-bentoml-production/06-production-deployment.md)