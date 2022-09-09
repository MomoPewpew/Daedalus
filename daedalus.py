import argparse
import os

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

    os.system("sudo shutdown -P +10")

    os.system("rm /home/ubuntu/Daedalus/in/*")
    os.system("rm /home/ubuntu/Daedalus/out/*")

    eval(opt.function + '(opt)')

    ##TODO: Start shutdown countdown?

def arcanegan(opt: argparse.Namespace) -> None:
    if opt.sourceURL == "":
        return
    
    os.system(f"wget -P /home/ubuntu/Daedalus/in {opt.sourceURL}")

    cmd = 'conda run -n arcanegan python3 /home/ubuntu/Daedalus/plugins/ArcaneGAN/arcanegan.py'
    handlecmd(cmd)

def handlecmd(cmd: str) -> None:
    os.system(cmd)
    os.system("sudo shutdown -P +1")

if __name__ == "__main__":
    main()