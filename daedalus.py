import argparse
import glob
import os
from PIL import Image

def main() -> None:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--function",
        type=str,
        default="",
        help="What function we are using"
    )
    parser.add_argument(
        "--sourceURL",
        type=str,
        default="",
        help="The URL to the source that we need to download."
    )
    parser.add_argument(
        "--args",
        type=str,
        default="",
        help="The arguments that will be sent to the called function"
    )

    opt = parser.parse_args()

    if opt.function == "":
        return
    
    opt.args = opt.args.replace("#arg#", "--").replace("#qt#", "\"")

    os.system("sudo shutdown -P +10")

    os.system("rm /home/ubuntu/Daedalus/in/*")
    os.system("rm /home/ubuntu/Daedalus/out/*")

    cmd = eval(opt.function + '(opt)')

    if cmd != "":
        os.system(cmd)

    os.system("sudo shutdown -P +1")

def download(sourceURL: str) -> bool:
    if sourceURL == "":
        return False

    os.system(f"wget -P /home/ubuntu/Daedalus/in {sourceURL}")
    return True

def convertInToJPG() -> None:
    filenames = glob.glob("/home/ubuntu/Daedalus/in/*")

    assert os.path.isfile(filenames[0])

    if ".jpg" in filenames[0]: return

    path, name = os.path.split(filenames[0])
    index = name.find(".")
    baseName = name[:index]

    Image.open(filenames[0]).convert("RGB").save(f'{path}/{baseName}.jpg')
    os.system(f"rm {filenames[0]}")

def arcanegan(opt: argparse.Namespace) -> str:
    if download(opt.sourceURL):
        return 'conda run -n arcanegan python3 /home/ubuntu/Daedalus/plugins/ArcaneGAN/arcanegan.py'

    return ""

def txt2imgSingle(opt: argparse.Namespace) -> str:
    return f"conda run -n stablediffusion python3 /home/ubuntu/Daedalus/plugins/stable-diffusion/scripts/txt2imgSingle.py {opt.args}"

def txt2imgBatch(opt: argparse.Namespace) -> str:
    return f"conda run -n stablediffusion python3 /home/ubuntu/Daedalus/plugins/stable-diffusion/scripts/txt2imgBatch.py {opt.args}"

def txt2imgVariations(opt: argparse.Namespace) -> str:
    return f"conda run -n stablediffusion python3 /home/ubuntu/Daedalus/plugins/stable-diffusion/scripts/txt2imgVariations.py {opt.args}"

def img2imgSingle(opt: argparse.Namespace) -> str:
    if download(opt.sourceURL):
        return f"conda run -n stablediffusion python3 /home/ubuntu/Daedalus/plugins/stable-diffusion/scripts/img2imgSingle.py {opt.args}"

    return ""

def img2imgBatch(opt: argparse.Namespace) -> str:
    if download(opt.sourceURL):
        return f"conda run -n stablediffusion python3 /home/ubuntu/Daedalus/plugins/stable-diffusion/scripts/img2imgBatch.py {opt.args}"

    return ""

def img2imgVariations(opt: argparse.Namespace) -> str:
    if download(opt.sourceURL):
        return f"conda run -n stablediffusion python3 /home/ubuntu/Daedalus/plugins/stable-diffusion/scripts/img2imgVariations.py {opt.args}"

    return ""

def realesrgan(opt: argparse.Namespace) -> str:
    if download(opt.sourceURL):
        return 'conda run -n realesrgan python3 /home/ubuntu/Daedalus/plugins/Real-ESRGAN/upscale_image.py'

    return ""

def gfpgan(opt: argparse.Namespace) -> str:
    if download(opt.sourceURL):
        return 'conda run -n realesrgan python3 /home/ubuntu/Daedalus/plugins/Real-ESRGAN/upscale_image.py --face_enhance'

    return ""

def ThreeDInPaint(opt: argparse.Namespace) -> str:
    if download(opt.sourceURL):
        convertInToJPG()
        os.system("sudo shutdown -P +20")
        return f'conda run -n 3DP python3 /home/ubuntu/Daedalus/plugins/3d-photo-inpainting/main.py {opt.args}'

    return ""

if __name__ == "__main__":
    main()