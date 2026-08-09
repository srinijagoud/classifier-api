from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id="Srinija2/sentiment-distilbert", repo_type="model")
api.upload_folder(
    folder_path="models/sentiment-distilbert",
    repo_id="Srinija2/sentiment-distilbert",
)