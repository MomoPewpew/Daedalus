import argparse
import asyncio
import os

taskID = -1

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--taskID",
        type=int,
        default="-1",
        help="The ID of the task that is being worked on."
    )
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

    if opt.sourceURL != "":
        pass
        ##TODO: Download source url

    taskID = opt.taskID

    eval(opt.function + '(opt.args)')

def arcanegan(args: str) -> None:
    cmd = 'conda run -n arcanegan python3 Daedalus/plugins/ArcaneGAN/arcanegan.py'
    handlecmd(cmd)

def handlecmd(cmd: str) -> None:
    os.system(cmd)

if __name__ == "__main__":
    main()