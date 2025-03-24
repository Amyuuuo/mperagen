import wandb


def download_wandb(run_id, out_dir="./log", team="ragen", project="RAGEN"):
    api = wandb.Api()
    run = api.run(f"{team}/{project}/{run_id}")
    files = run.files()
    for file in files:
        file.download(out_dir + "/" + run_id, exist_ok=True)

# usage: 
# from ragen.utils.wandb import download_wandb
# download_wandb("9o465jqj")
