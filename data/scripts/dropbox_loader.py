import os
import click
import dropbox
from tqdm import tqdm

@click.command()
@click.argument("path", default=".")
@click.option("--api_token", default=os.environ.get('DROPBOX_TOKEN', ''))
@click.option("--target", default="/smth")
def upload_files(api_token, path, target="/smth"):
    if not os.path.exists(path):
        print("path doesn't exist", path)
        return
    if os.path.isdir(path):
        paths = [os.path.join(path, p) for p in os.listdir(path)]
    else:
        paths = [path]

    try:
        d = dropbox.Dropbox(api_token)
        for path in tqdm(paths):
            with open(path, "rb") as f:
                print(path)
                # upload gives you metadata about the file
                # we want to overwite any previous version of the file
                targetfile = os.path.join(target, os.path.basename(path))
                meta = d.files_upload(f.read(), targetfile, mode=dropbox.files.WriteMode("overwrite"))
                print(meta)
    except Exception as e:
        print("exception while uploading", e)

if __name__ == "__main__":
    upload_files()
