from athena import gpu_ops as ad

import os
import sys
import yaml
import json
import multiprocessing
import signal


def main():
    def start_scheduler(settings):
        for key, value in settings.items():
            os.environ[key] = str(value)
        assert os.environ['DMLC_ROLE'] == "scheduler"
        print('Scheduler starts...')
        ad.scheduler_init()
        ad.scheduler_finish()
    def signal_handler(sig, frame):
        print("SIGINT signal caught, stop Training")
        proc.kill()
        exit(0)

    if len(sys.argv) == 1:
        settings = json.load(open('./settings/scheduler.json'))
    else:
        file_path = sys.argv[1]
        suffix = file_path.split('.')[-1]
        if suffix == 'yml':
            settings = yaml.load(open(file_path).read(), Loader=yaml.FullLoader)
        elif suffix == 'json':
            settings = json.load(open(file_path))
        else:
            assert False, 'File type not supported.'    
    print('Scheduler settings:')
    print(settings)    

    proc = multiprocessing.Process(target=start_scheduler, args=(settings,))
    proc.start()
    signal.signal(signal.SIGINT, signal_handler)
    proc.join()


if __name__ == '__main__':
    main()
