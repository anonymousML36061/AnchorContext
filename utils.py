import os
import shutil

def get_checkpoint_list(checkpoint_dir):
    dir_steps_list = []
    dir_list = os.listdir(checkpoint_dir)
    if len(dir_list) > 0:
        for _dir in dir_list:
            completed_steps = int(_dir.split('_')[1])
            loader_steps = int(_dir.split('_')[2])
            dir_steps_list.append([_dir, completed_steps, loader_steps])
        dir_steps_list = sorted(dir_steps_list, key=lambda x: x[1])
    else:
        dir_steps_list = [['', 0, 0]]
    return dir_steps_list


def clean_checkpoint_folder(checkpoint_dir, max_keep=1):
    # assert max_keep > 0, "max_keep should be greater than 0"
    dir_steps_list = get_checkpoint_list(checkpoint_dir)
    if len(dir_steps_list) >= max_keep and dir_steps_list[0][0] != '':
        for _dir, _, _ in dir_steps_list[:len(dir_steps_list)-max_keep]:
            shutil.rmtree(os.path.join(checkpoint_dir, _dir))





import debugpy
from termcolor import colored

def setup_debugpy(accelerator, endpoint="localhost", port=5678, rank=0, force=False):
    if "DEBUGPY" not in os.environ:
        print(colored(f"DEBUGPY not in os.environ", "red"))
        return
    rank = int(os.getenv("DEBUGPY_RANK", rank))
    port = int(os.getenv("DEBUGPY_PORT", port))
    endpoint = os.getenv("DEBUGPY_ENDPOINT", endpoint)
    if accelerator.process_index != rank:
        accelerator.wait_for_everyone()
        return
    # print(colored(f"rank: {get_rank()}, is_main_process: {is_main_process()}", "red"))
    if force:
        # run_cmd("ps aux | grep debugpy | awk '{print $2}' | xargs kill -9", fault_tolerance=True)
        print(debugpy(f"Force killed debugpy", "red"))
    try:
        debugpy.listen((endpoint, port))
        print(colored(f"Waiting for debugger attach on {endpoint}:{port}", "red"))
        debugpy.wait_for_client()
    except:
        print(colored(f"Failed to setup debugpy, {endpoint}:{port} occupied", "red"))

    accelerator.wait_for_everyone()
