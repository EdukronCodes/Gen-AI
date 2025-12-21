@echo off
REM Batch script to create GenAI Project folder structure with all files

set ROOT=genai_project

REM Root folder
mkdir %ROOT%

REM ================== Config ==================
mkdir %ROOT%\config
type nul > %ROOT%\config\settings.yaml
type nul > %ROOT%\config\prompts.yaml
type nul > %ROOT%\config\vectorstore.yaml
type nul > %ROOT%\config\evaluation.yaml
type nul > %ROOT%\config\logging.yaml

REM ================== Data ==================
mkdir %ROOT%\data\raw
mkdir %ROOT%\data\interim
mkdir %ROOT%\data\processed
mkdir %ROOT%\data\embeddings
mkdir %ROOT%\data\outputs

REM ================== Source Code ==================
mkdir %ROOT%\src\loaders
type nul > %ROOT%\src\loaders\pdf_loader.py
type nul > %ROOT%\src\loaders\slack_loader.py
type nul > %ROOT%\src\loaders\db_loader.py
type nul > %ROOT%\src\loaders\api_loader.py
type nul > %ROOT%\src\loaders\image_loader.py

mkdir %ROOT%\src\preprocessing
type nul > %ROOT%\src\preprocessing\cleaner.py
type nul > %ROOT%\src\preprocessing\splitter.py
type nul > %ROOT%\src\preprocessing\embedder.py
type nul > %ROOT%\src\preprocessing\metadata_handler.py

mkdir %ROOT%\src\retriever
type nul > %ROOT%\src\retriever\vectorstore.py
type nul > %ROOT%\src\retriever\hybrid_retriever.py
type nul > %ROOT%\src\retriever\retriever.py

mkdir %ROOT%\src\llm
type nul > %ROOT%\src\llm\prompts.py
type nul > %ROOT%\src\llm\chain.py
type nul > %ROOT%\src\llm\agent.py

mkdir %ROOT%\src\llm\tools
type nul > %ROOT%\src\llm\tools\search_tool.py
type nul > %ROOT%\src\llm\tools\calculator_tool.py
type nul > %ROOT%\src\llm\tools\db_query_tool.py

mkdir %ROOT%\src\llm\finetune
type nul > %ROOT%\src\llm\finetune\dataset_prep.py
type nul > %ROOT%\src\llm\finetune\trainer.py
type nul > %ROOT%\src\llm\finetune\evaluator.py

mkdir %ROOT%\src\pipelines
type nul > %ROOT%\src\pipelines\rag_pipeline.py
type nul > %ROOT%\src\pipelines\finetune_pipeline.py
type nul > %ROOT%\src\pipelines\multimodal_pipeline.py

mkdir %ROOT%\src\evaluation
type nul > %ROOT%\src\evaluation\metrics.py
type nul > %ROOT%\src\evaluation\ragas_eval.py
type nul > %ROOT%\src\evaluation\human_eval.py
type nul > %ROOT%\src\evaluation\hallucination_check.py

mkdir %ROOT%\src\monitoring
type nul > %ROOT%\src\monitoring\telemetry.py
type nul > %ROOT%\src\monitoring\logger.py
type nul > %ROOT%\src\monitoring\drift_detection.py

mkdir %ROOT%\src\app
type nul > %ROOT%\src\app\api.py
type nul > %ROOT%\src\app\ui_streamlit.py
type nul > %ROOT%\src\app\cli.py

mkdir %ROOT%\src\app\dashboards
type nul > %ROOT%\src\app\dashboards\logs_dashboard.py
type nul > %ROOT%\src\app\dashboards\eval_dashboard.py

REM ================== Scripts ==================
mkdir %ROOT%\scripts
type nul > %ROOT%\scripts\ingest_data.py
type nul > %ROOT%\scripts\run_inference.py
type nul > %ROOT%\scripts\retrain_embeddings.py
type nul > %ROOT%\scripts\export_results.py

REM ================== Tests ==================
mkdir %ROOT%\tests
type nul > %ROOT%\tests\test_loaders.py
type nul > %ROOT%\tests\test_retriever.py
type nul > %ROOT%\tests\test_llm.py
type nul > %ROOT%\tests\test_pipeline.py
type nul > %ROOT%\tests\test_eval.py

REM ================== Notebooks ==================
mkdir %ROOT%\notebooks
type nul > %ROOT%\notebooks\01_data_explore.ipynb
type nul > %ROOT%\notebooks\02_embedding_eval.ipynb
type nul > %ROOT%\notebooks\03_prompt_tuning.ipynb
type nul > %ROOT%\notebooks\04_rag_pipeline_dev.ipynb
type nul > %ROOT%\notebooks\05_eval_metrics.ipynb

REM ================== Docker ==================
mkdir %ROOT%\docker
type nul > %ROOT%\docker\Dockerfile
type nul > %ROOT%\docker\docker-compose.yml

REM ================== Deployment ==================
mkdir %ROOT%\deployment\k8s
type nul > %ROOT%\deployment\k8s\deployment.yaml
type nul > %ROOT%\deployment\k8s\service.yaml
type nul > %ROOT%\deployment\k8s\ingress.yaml

mkdir %ROOT%\deployment\terraform

mkdir %ROOT%\deployment\ci_cd
type nul > %ROOT%\deployment\ci_cd\pipeline.yaml

REM ================== MLflow ==================
mkdir %ROOT%\mlflow
type nul > %ROOT%\mlflow\tracking_server_config.yaml

REM ================== Root files ==================
type nul > %ROOT%\requirements.txt
type nul > %ROOT%\requirements-dev.txt
type nul > %ROOT%\README.md
type nul > %ROOT%\.env
type nul > %ROOT%\.gitignore

echo ✅ GenAI Project structure with all files created successfully!
pause
